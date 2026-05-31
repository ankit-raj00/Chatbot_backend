import logging
import time
from rag.graph.state import RAGGraphState
from rag.vector_store.qdrant_manager import QdrantManager
from qdrant_client import models
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
import structlog
logger = structlog.get_logger(__name__)


def _invoke_with_retry(retriever, question: str):
    """
    Invoke retriever with exponential backoff + jitter for 429 RESOURCE_EXHAUSTED.

    Two-layer retry strategy (per Google's recommendation):
      Layer 1 — SDK: GoogleGenerativeAIEmbeddings(max_retries=6) in QdrantManager
      Layer 2 — Here: outer retry with jitter for bursts that exhaust the SDK retries.

    Uses tenacity if available, falls back to manual backoff.
    """
    try:
        from tenacity import (
            retry, stop_after_attempt, wait_exponential_jitter,
            retry_if_exception_message, before_sleep_log
        )

        @retry(
            retry=retry_if_exception_message(match=r".*(429|RESOURCE_EXHAUSTED|resource exhausted).*"),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=5, max=60, jitter=3),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _do():
            return retriever.invoke(question)

        return _do()

    except ImportError:
        # Fallback: simple manual backoff if tenacity not installed
        delay = 5
        for attempt in range(1, 5):
            try:
                return retriever.invoke(question)
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
                if is_rate_limit and attempt < 4:
                    logger.warning(
                        f"⏳ Google 429 (attempt {attempt}/4). Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise


class RetrievalNode:
    """
    The 'Retriever' node responsible for fetching relevant documents.
    Implements Similarity Search with optional file-level filtering.
    """

    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.vector_store = self.qdrant_manager.get_vector_store()

    def retrieve(self, state: RAGGraphState) -> RAGGraphState:
        """
        Retrieves documents based on the current question.

        File filtering uses metadata.source, which requires a Qdrant payload
        index — created automatically in QdrantManager.ensure_collection().
        Without the index, Qdrant Cloud returns:
          400 "Index required but not found for metadata.source [keyword]"
        """
        question = state["question"]
        logger.info(f"Retrieving for question: {question}")

        search_strategy = "similarity"
        logger.info(f"   🔍 Search: [strategy={search_strategy}] [k=5]")

        search_kwargs = {"k": 5}

        # ── File-level filtering by UUID (needs payload index on metadata.file_id) ──
        # Filtering by file_id (UUID) not filename:
        #   - Unique per upload even if same filename is re-uploaded
        #   - Isolated per user — no cross-user leakage
        selected_file_ids = state.get("selected_file_ids")
        if selected_file_ids and isinstance(selected_file_ids, list) and len(selected_file_ids) > 0:
            logger.info(f"   🛡️ File Filter (by UUID): {selected_file_ids}")
            file_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_id",      # UUID stored at ingestion time
                        match=models.MatchAny(any=selected_file_ids)
                    )
                ]
            )
            search_kwargs["filter"] = file_filter

        retriever = self.vector_store.as_retriever(
            search_type=search_strategy,
            search_kwargs=search_kwargs
        )

        try:
            # Retry handles Google 429 RESOURCE_EXHAUSTED (two-layer defence)
            documents = _invoke_with_retry(retriever, question)
            logger.info(f"   ✅ Retrieved {len(documents)} documents.")

            return {
                "question": question,
                "documents": documents,
                "generation": state.get("generation"),
                "web_search_needed": False,
                "hallucination_count": state.get("hallucination_count", 0),
                "retry_count": state.get("retry_count", 0)
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return {
                **state,
                "documents": []
            }

async def parallel_retrieve_node(state: RAGGraphState) -> RAGGraphState:
    """
    Runs Qdrant vector search and Tavily web search in parallel.
    Merges results, deduplicates by content hash.
    Falls back gracefully if either source fails.

    WHY asyncio.gather with return_exceptions=True:
        If Tavily is down or key is missing, we still get Qdrant results.
        The exception is caught per-source, not globally.
    """
    import asyncio
    import hashlib
    from langchain_core.documents import Document

    question = state["question"]
    selected_file_ids = state.get("selected_file_ids")

    # ── Task 1: Qdrant vector search ─────────────────────────────
    async def run_qdrant():
        node = RetrievalNode()
        # Use asyncio.to_thread because retrieve() is synchronous
        result_state = await asyncio.to_thread(node.retrieve, {
            "question": question,
            "selected_file_ids": selected_file_ids,
            "documents": [], "generation": None,
            "web_search_needed": False,
            "hallucination_count": 0, "retry_count": 0, "messages": []
        })
        return result_state.get("documents", [])

    # ── Task 2: Tavily web search ─────────────────────────────────
    async def run_web_search():
        from langchain_tavily import TavilySearch
        from langchain_core.documents import Document
        try:
            tool = TavilySearch(max_results=3)
            results = tool.invoke({"query": question})
            docs = []
            if isinstance(results, list):
                for r in results:
                    docs.append(Document(
                        page_content=r.get("content", ""),
                        metadata={"source": r.get("url", "web_search"), "type": "web"}
                    ))
            return docs
        except Exception as e:
            logger.warning(f"Parallel web search failed: {e}")
            return []

    # ── Run both in parallel ──────────────────────────────────────
    qdrant_docs, web_docs = await asyncio.gather(
        run_qdrant(),
        run_web_search(),
        return_exceptions=True
    )

    # Handle exceptions from gather
    if isinstance(qdrant_docs, Exception):
        logger.error(f"Qdrant retrieval failed in parallel mode: {qdrant_docs}")
        qdrant_docs = []
    if isinstance(web_docs, Exception):
        logger.warning(f"Web search failed in parallel mode: {web_docs}")
        web_docs = []

    # ── Merge and deduplicate ─────────────────────────────────────
    all_docs = list(qdrant_docs) + list(web_docs)

    seen_hashes = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    logger.info(
        f"Parallel retrieval: {len(qdrant_docs)} vector + {len(web_docs)} web = "
        f"{len(unique_docs)} unique docs"
    )

    return {
        **state,
        "question": question,
        "documents": unique_docs,
        "web_search_needed": False,   # Already did web search in parallel
        "hallucination_count": state.get("hallucination_count", 0),
        "retry_count": state.get("retry_count", 0),
    }
