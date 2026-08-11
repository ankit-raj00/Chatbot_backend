import os
import re
import shutil
import uuid
import random
import asyncio
import logging
from typing import List, Dict, Any
from fastapi import UploadFile

# try:
#     from langchain.retrievers import ParentDocumentRetriever
# except ImportError:
#     from langchain_community.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

# Placeholder for now
ParentDocumentRetriever = None

from langchain_core.stores import InMemoryStore # Restored Import

from rag.ingestion_router import IngestionRouter
from rag.parsers.llama_parse_client import LlamaParseClient
from rag.parsers.parser_client import parse_pdf_blocks, ParserServiceError
from rag.parsers.native_blocks import (
    blocks_from_file as native_blocks_from_file,
    SUPPORTED_EXTS as NATIVE_SUPPORTED_EXTS,
)
from rag.chunking.splitter_factory import SplitterFactory
from rag.chunking.block_chunker import chunk_blocks
from rag.chunking.doc_router import classify_doc_type, profile_for
from rag.vector_store.qdrant_manager import QdrantManager
from services.ingestion_job_service import IngestionJobService

# Use the hybrid parser microservice for PDFs (structure-aware blocks).
USE_PARSER_SERVICE = os.getenv("USE_PARSER_SERVICE", "true").lower() == "true"
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
# Google's free embedding tier is 100 requests/min. Strategy:
#  - proactive pacing: add in throttled batches (EMBED_BATCH / EMBED_BATCH_SLEEP)
#  - reactive retry: on 429 RESOURCE_EXHAUSTED, back off (honoring the API's
#    retryDelay) and retry the batch, so a transient limit hit never fails ingest.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "80"))
EMBED_BATCH_SLEEP = float(os.getenv("EMBED_BATCH_SLEEP", "60"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "8"))
EMBED_MAX_BACKOFF = float(os.getenv("EMBED_MAX_BACKOFF", "65"))


_CHUNK_ID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _chunk_id(file_id: str, idx: int) -> str:
    """Deterministic point id → retries overwrite the same point (idempotent, no dupes)."""
    return str(uuid.uuid5(_CHUNK_ID_NS, f"{file_id}:{idx}"))


def _is_retryable_error(msg: str) -> bool:
    """Rate-limit OR transient infra errors worth retrying (idempotent ids make this safe)."""
    m = msg.lower()
    return any(k in m for k in (
        "resource_exhausted", "429", "quota", "rate limit", "rate_limit",   # rate limit
        "timed out", "timeout", "temporarily unavailable", "unavailable",   # transient
        "503", "502", "connection", "reset by peer",
    ))


def _parse_retry_delay(msg: str) -> "float | None":
    """Pull the server-suggested wait (seconds) out of a Google 429 message."""
    # e.g. "Please retry in 3.904844898s"
    m = re.search(r"retry in ([\d.]+)s", msg)
    if m:
        return float(m.group(1))
    # e.g. 'retryDelay': '3s'
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg)
    if m:
        return float(m.group(1))
    return None

logging.basicConfig(level=logging.INFO)
import structlog
logger = structlog.get_logger(__name__)

class IngestionService:
    """
    Orchestrates the RAG Ingestion Pipeline.
    Connects Router -> Parser -> Splitter -> Vector Store.
    """
    
    def __init__(self):
        self.router = IngestionRouter()
        self.parser = LlamaParseClient()
        self.qdrant_manager = QdrantManager()
        self.splitter_factory = SplitterFactory()
        
        # Parent Document Store
        # Using InMemoryStore to avoid 'LocalFileStore' import issues on some environments.
        # Note: In Prod, use RedisStore or similar for persistence.
        self.parent_store = InMemoryStore()

    async def process_upload(self, file: UploadFile, document_type: str = "Auto (Detect)", user_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for file ingestion.
        Args:
            file: The uploaded file object
            document_type: Manual category override (optional)
            user_id: The ID of the user uploading the file (for isolation)
        """
        temp_path = ""
        try:
            logger.info(f"--- 📥 Starting Ingestion for: {file.filename} [Type: {document_type}] ---")

            # 1. Save to Temp File
            file_ext = os.path.splitext(file.filename)[1]
            temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
            temp_path = os.path.join("temp", temp_filename)
            
            if not os.path.exists("temp"):
                os.makedirs("temp")
                
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(temp_path)
            
            # 2. Smart Routing 🧠
            logger.info(f"   🚦 Routing... (Size: {file_size} bytes)")
            route_config = self.router.route(file.filename, file.content_type, file_size, force_category=document_type)
            logger.info(f"   ✅ Route Decided: [{route_config['type_category']}]")
            logger.info(f"      Parser: {route_config['parser_strategy']} | Chunking: {route_config['chunking_strategy']}")
            logger.info(f"      Rationale: {route_config['rationale']}")
            
            # 3. Parsing (Eyes) 👀
            # Note: route_config['parser_config'] is passed to the parser
            logger.info("   👀 Parsing Document...")
            docs = await self.parser.parse(temp_path, route_config["parser_config"])
            logger.info(f"   ✅ Parsed {len(docs)} raw documents.")
            
            # CRITICAL: Generate a stable UUID for this upload.
            # Use this as the filter key — NOT the filename, which can clash
            # across users or repeated uploads of the same file.
            file_id = str(uuid.uuid4())

            # Tag every chunk with:
            #   metadata.source  → original filename (for display)
            #   metadata.file_id → UUID (for Qdrant filtering)
            #   metadata.user_id → User ID (for user isolation)
            for doc in docs:
                doc.metadata["source"]  = file.filename  # display label
                doc.metadata["file_id"] = file_id        # filter key
                if user_id:
                    doc.metadata["user_id"] = user_id    # isolate to user
            
            # 4. Splitting & Indexing (Brain & Memory) 🧠 + 💾
            logger.info("   🔪 Splitting & Indexing...")
            await self._index_documents(docs, route_config)
            logger.info("   💾 Indexing Complete.")
            
            # 5. Cleanup
            os.remove(temp_path)
            
            logger.info("--- ✅ Ingestion Finished Successfully ---")
            
            return {
                "status": "success",
                "file_id": file_id,          # UUID — use this for filtering
                "filename": file.filename,   # original name — for display only
                "strategy": route_config,
                "chunks_processed": "dynamic"
            }
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {str(e)}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    async def _update_job(self, job_id: str, **fields) -> None:
        """Best-effort progress update — a Redis hiccup here must never fail ingestion."""
        if not job_id:
            return
        try:
            await IngestionJobService.update_job(job_id, **fields)
        except Exception as e:
            logger.warning(f"   (progress update failed, ignoring: {e})")

    async def process_upload_from_path(
        self,
        file_path: str,
        filename: str,
        document_type: str = "Auto (Detect)",
        user_id: str = None,
        job_id: str = None,
    ) -> dict:
        """
        Processes an already-saved file at file_path.
        Called by the background job task — the UploadFile is no longer available.
        Same pipeline as process_upload but accepts a path instead of UploadFile.
        Returns: {"status", "file_id", "filename", "strategy", "chunks_count"}

        job_id, if given, gets its progress_message updated live as each stage
        actually starts/finishes — so the status the frontend shows reflects
        which parser really ran, instead of a fixed guess made before routing.
        """
        import uuid as uuid_mod
        from langchain_core.documents import Document as LCDoc

        try:
            logger.info(f"--- 📥 Background Ingestion: {filename} [Type: {document_type}] ---")

            file_size = os.path.getsize(file_path)

            # Smart Routing
            route_config = self.router.route(filename, "", file_size, force_category=document_type)
            # keep the raw user selection so block-based classification can honour
            # an explicit choice but self-classify when it's "Auto (Detect)".
            route_config["_forced_document_type"] = document_type
            logger.info(f"   ✅ Route: [{route_config['type_category']}] — {route_config['rationale']}")

            file_id = str(uuid_mod.uuid4())

            # ── Structure-aware path: PDFs go to the parser microservice, which
            # returns typed blocks; we chunk on structure (headings/callouts).
            if USE_PARSER_SERVICE and filename.lower().endswith(".pdf"):
                await self._update_job(job_id, progress_message="Parsing with custom layout parser...")
                try:
                    chunks_count = await self._ingest_pdf_blocks(
                        file_path, filename, file_id, user_id, route_config
                    )
                    logger.info(f"--- ✅ Block Ingestion Complete: {filename} ({chunks_count} chunks) ---")
                    return {
                        "status": "success",
                        "file_id": file_id,
                        "filename": filename,
                        "strategy": {**route_config, "parser_strategy": "parser_service",
                                     "chunking_strategy": "structure_blocks"},
                        "chunks_count": chunks_count,
                    }
                except ParserServiceError as e:
                    logger.warning(f"   ⚠️ Parser service unavailable ({e}); falling back to LlamaParse")
                    await self._update_job(
                        job_id,
                        progress_message="Custom parser unavailable, retrying with LlamaParse...",
                    )

            # ── Native non-PDF path: docx/pptx/xlsx/csv/text → typed blocks → same
            # structure-aware pipeline as PDFs (routing/chunking/hybrid/rerank).
            elif os.path.splitext(filename)[1].lower() in NATIVE_SUPPORTED_EXTS:
                await self._update_job(job_id, progress_message="Extracting document structure...")
                nblocks = await asyncio.to_thread(native_blocks_from_file, file_path, filename)
                if nblocks:
                    chunks_count = await self._ingest_blocks(
                        nblocks, filename, file_id, user_id, route_config, parser_label="native"
                    )
                    logger.info(f"--- ✅ Native Block Ingestion Complete: {filename} ({chunks_count} chunks) ---")
                    return {
                        "status": "success",
                        "file_id": file_id,
                        "filename": filename,
                        "strategy": {**route_config, "parser_strategy": "native",
                                     "chunking_strategy": "structure_blocks"},
                        "chunks_count": chunks_count,
                    }
                logger.warning(f"   ⚠️ Native extraction yielded no blocks for {filename}; falling back to LlamaParse")
                await self._update_job(
                    job_id,
                    progress_message="No native structure found, retrying with LlamaParse...",
                )

            # Parsing (fallback / unsupported)
            await self._update_job(job_id, progress_message="Parsing document with LlamaParse...")
            docs = await self.parser.parse(file_path, route_config["parser_config"])
            logger.info(f"   ✅ Parsed {len(docs)} raw documents.")

            for doc in docs:
                doc.metadata["source"]  = filename
                doc.metadata["file_id"] = file_id
                if user_id:
                    doc.metadata["user_id"] = user_id

            # Splitting & Indexing
            await self._index_documents(docs, route_config)

            # Estimate chunks count from standard split
            try:
                splitter = self.splitter_factory.get_splitter(
                    route_config.get("chunking_strategy", "recursive"),
                    route_config.get("chunker_config", {})
                )
                chunks = splitter.split_documents(docs)
                chunks_count = len(chunks)
            except Exception:
                chunks_count = len(docs)

            logger.info(f"--- ✅ Background Ingestion Complete: {filename} ({chunks_count} chunks) ---")

            return {
                "status": "success",
                "file_id": file_id,
                "filename": filename,
                "strategy": route_config,
                "chunks_count": chunks_count,
            }

        except Exception as e:
            logger.error(f"❌ Background ingestion failed: {e}", exc_info=True)
            raise

    async def _ingest_pdf_blocks(
        self,
        file_path: str,
        filename: str,
        file_id: str,
        user_id: str,
        route_config: Dict[str, Any],
    ) -> int:
        """
        PDF → parser microservice (typed blocks) → structure-aware chunks → Qdrant.

        Chunks respect document structure: they never straddle a heading, and
        callouts/tables/figures stay atomic. Each chunk carries its heading
        breadcrumb + page so retrieval can cite and the agent gets context.
        """
        result = await parse_pdf_blocks(file_path, mode=os.getenv("PARSER_MODE", "auto"))
        blocks = result.get("blocks") or []
        if not blocks:
            raise ParserServiceError("parser returned no blocks")
        return await self._ingest_blocks(
            blocks, filename, file_id, user_id, route_config, parser_label="parser_service"
        )

    async def _ingest_blocks(
        self,
        blocks: List[Dict[str, Any]],
        filename: str,
        file_id: str,
        user_id: str,
        route_config: Dict[str, Any],
        parser_label: str = "native",
    ) -> int:
        """
        Shared core: typed blocks → doc-type routing → structure-aware chunks →
        throttled/retry embed → Qdrant. Used by both the PDF parser path and the
        native (docx/pptx/xlsx/csv/text) path so they behave identically.
        """
        # Doc-type routing → chunker profile. A user-selected document_type (via
        # route_config) overrides; otherwise classify from the blocks + filename.
        forced = route_config.get("_forced_document_type") or route_config.get("type_category")
        doc_type = classify_doc_type(blocks, filename, forced=forced)
        profile = profile_for(doc_type)
        logger.info(f"   🧭 doc_type={doc_type} → chunker {profile}")

        chunks = chunk_blocks(
            blocks,
            base_metadata={
                "source": filename,
                "file_id": file_id,
                "user_id": user_id,
                "doc_type": doc_type,
                "parser": parser_label,
            },
            **profile,
        )
        logger.info(f"   🔪 {len(blocks)} blocks → {len(chunks)} structure-aware chunks ({doc_type})")

        docs: List[Document] = []
        for c in chunks:
            meta = dict(c["metadata"])
            # Qdrant payload indexes work best on scalars — keep a flat copy too.
            hp = meta.get("heading_path") or []
            meta["heading_path"] = [h for h in hp if h]
            meta["section_path"] = " > ".join(meta["heading_path"])
            docs.append(Document(page_content=c["text"], metadata=meta))

        # deterministic ids so any retry (rate-limit or transient) is idempotent
        ids = [_chunk_id(file_id, k) for k in range(len(docs))]
        vectorstore = self.qdrant_manager.get_vector_store()
        # Throttled batches (proactive) + retry-with-backoff (reactive) to survive
        # the free-tier rate limit. On unrecoverable failure, roll back everything
        # for this file_id — a partial write orphans chunks that then surface as
        # duplicates in retrieval.
        try:
            for i in range(0, len(docs), EMBED_BATCH):
                batch = docs[i:i + EMBED_BATCH]
                await self._add_documents_with_retry(vectorstore, batch, ids[i:i + EMBED_BATCH])
                done = min(i + EMBED_BATCH, len(docs))
                logger.info(f"   💾 embedded {done}/{len(docs)} chunks")
                if done < len(docs):
                    await asyncio.sleep(EMBED_BATCH_SLEEP)
        except Exception:
            logger.warning(f"   ↩️ indexing failed — rolling back chunks for file_id={file_id}")
            try:
                await asyncio.to_thread(self.qdrant_manager.delete_file, file_id, user_id)
            except Exception as cleanup_err:
                logger.error(f"   rollback failed: {cleanup_err}")
            raise
        return len(docs)

    async def _add_documents_with_retry(self, vectorstore, batch, ids=None) -> None:
        """
        Embed + index one batch, retrying on rate-limit (429 RESOURCE_EXHAUSTED)
        or transient infra errors with exponential backoff that honors the
        server's suggested retryDelay. Deterministic ids make retries idempotent
        (no duplicates even if a timed-out write actually landed). Non-retryable
        errors are raised immediately (caller rolls back).
        """
        backoff = 5.0
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                # batch_size == len(batch) → one embed+upsert unit; with fixed ids
                # a retry overwrites the same points rather than duplicating them.
                await asyncio.to_thread(
                    vectorstore.add_documents, batch, ids=ids, batch_size=len(batch)
                )
                return
            except Exception as e:
                msg = str(e)
                if not _is_retryable_error(msg) or attempt == EMBED_MAX_RETRIES:
                    raise
                wait = _parse_retry_delay(msg) or backoff
                wait = min(wait, EMBED_MAX_BACKOFF) + random.uniform(0.5, 2.5)  # jitter
                logger.warning(
                    f"   ⏳ embedding rate-limited (attempt {attempt}/{EMBED_MAX_RETRIES}) "
                    f"— backing off {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, EMBED_MAX_BACKOFF)

    async def _index_documents(self, docs: List[Document], config: Dict[str, Any]):
        """
        Handles the Chunking and Indexing logic based on strategy.
        """
        chunk_strategy = config["chunking_strategy"]
        chunker_config = config["chunker_config"]

        vectorstore = self.qdrant_manager.get_vector_store()

        # --- Strategy A: Parent Document Retrieval (Finance/Medical) ---
        if chunk_strategy == "parent_document":
            # ParentDocumentRetriever needs two splitters:
            # 1. Child Splitter (Small chunks for vector search)
            # 2. Parent Splitter (Optional, usually keeping the whole doc or large chunks)

            if not ParentDocumentRetriever:
                logger.warning("ParentDocumentRetriever not available. Falling back to standard indexing.")
                # Fallback to standard strategy if ParentDocumentRetriever is not enabled
                await self._process_standard_strategy(docs, config)
                return

            # The following code is unreachable until ParentDocumentRetriever is enabled
            child_splitter = self.splitter_factory.get_splitter(
                "recursive", 
                {"chunk_size": chunker_config.get("child_chunk_size", 200)}
            )
            
            # For Parent, we either keep the whole doc (if small) or split into large chunks
            parent_splitter = self.splitter_factory.get_splitter(
                "recursive",
                {"chunk_size": chunker_config.get("parent_chunk_size", 4000)}
            )
            
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=self.parent_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            
            retriever.add_documents(docs)
            logger.info("Indexed using ParentDocumentRetriever")
            
        # --- Strategy B: Standard Vector Search (Recursive / Semantic / etc) ---
        else:
            await self._process_standard_strategy(docs, config)

    async def _process_standard_strategy(self, docs: List[Document], config: Dict[str, Any]):
        """
        Helper to run standard chunking and indexing (Strategy B).
        Used as fallback for Strategy A or directly for Strategy B.

        Throttled batches + retry-with-backoff — same pattern as
        _ingest_blocks/_add_documents_with_retry. This used to call
        vectorstore.add_documents(chunks) in one unthrottled shot, so any
        document producing enough chunks to cross Google's free-tier
        embedding quota (100 req/min) crashed ingestion outright instead of
        backing off — confirmed live with a heavily-illustrated PDF.
        """
        chunk_strategy = config["chunking_strategy"]
        chunker_config = config["chunker_config"]
        vectorstore = self.qdrant_manager.get_vector_store()

        # 1. Split
        splitter = self.splitter_factory.get_splitter(chunk_strategy, chunker_config)
        chunks = splitter.split_documents(docs)

        # Deterministic ids (same scheme as _ingest_blocks) so a retry
        # overwrites the same points instead of duplicating them. Both
        # callers tag doc.metadata["file_id"] before splitting, and
        # LangChain's splitters propagate source metadata onto each chunk.
        ids = [_chunk_id(c.metadata.get("file_id") or "unknown", i) for i, c in enumerate(chunks)]

        # 2. Index — throttled batches + retry-with-backoff.
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            await self._add_documents_with_retry(vectorstore, batch, ids[i:i + EMBED_BATCH])
            done = min(i + EMBED_BATCH, len(chunks))
            logger.info(f"   💾 embedded {done}/{len(chunks)} chunks (standard strategy)")
            if done < len(chunks):
                await asyncio.sleep(EMBED_BATCH_SLEEP)

        logger.info(f"Indexed {len(chunks)} chunks using Standard Strategy (Fallback/Direct)")
