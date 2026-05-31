import os
import shutil
import uuid
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
from rag.chunking.splitter_factory import SplitterFactory
from rag.vector_store.qdrant_manager import QdrantManager

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
            self._index_documents(docs, route_config)
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

    async def process_upload_from_path(
        self,
        file_path: str,
        filename: str,
        document_type: str = "Auto (Detect)",
        user_id: str = None
    ) -> dict:
        """
        Processes an already-saved file at file_path.
        Called by the background job task — the UploadFile is no longer available.
        Same pipeline as process_upload but accepts a path instead of UploadFile.
        Returns: {"status", "file_id", "filename", "strategy", "chunks_count"}
        """
        import uuid as uuid_mod
        from langchain_core.documents import Document as LCDoc

        try:
            logger.info(f"--- 📥 Background Ingestion: {filename} [Type: {document_type}] ---")

            file_size = os.path.getsize(file_path)

            # Smart Routing
            route_config = self.router.route(filename, "", file_size, force_category=document_type)
            logger.info(f"   ✅ Route: [{route_config['type_category']}] — {route_config['rationale']}")

            # Parsing
            docs = await self.parser.parse(file_path, route_config["parser_config"])
            logger.info(f"   ✅ Parsed {len(docs)} raw documents.")

            file_id = str(uuid_mod.uuid4())

            for doc in docs:
                doc.metadata["source"]  = filename
                doc.metadata["file_id"] = file_id
                if user_id:
                    doc.metadata["user_id"] = user_id

            # Splitting & Indexing
            self._index_documents(docs, route_config)

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

    def _index_documents(self, docs: List[Document], config: Dict[str, Any]):
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
                # The original _index_documents is not async, so we call the standard strategy directly.
                # If _index_documents were async, this would be `return await self._process_standard_strategy(docs, config)`
                self._process_standard_strategy(docs, config)
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
            self._process_standard_strategy(docs, config)

    def _process_standard_strategy(self, docs: List[Document], config: Dict[str, Any]):
        """
        Helper to run standard chunking and indexing (Strategy B).
        Used as fallback for Strategy A or directly for Strategy B.
        """
        chunk_strategy = config["chunking_strategy"]
        chunker_config = config["chunker_config"]
        vectorstore = self.qdrant_manager.get_vector_store()
        
        # 1. Split
        splitter = self.splitter_factory.get_splitter(chunk_strategy, chunker_config)
        chunks = splitter.split_documents(docs)
        
        # 2. Index
        vectorstore.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks using Standard Strategy (Fallback/Direct)")
