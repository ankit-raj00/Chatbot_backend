import logging

from rag.graph.state import RAGGraphState
from rag.tools.web_search import search_web_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchNode:
    """
    The 'Web Search' node for fallback retrieval (TinyFish-backed).
    Implements Defense #4: Hybrid Search expansion (Internal + External).
    """

    def __init__(self):
        # No client to construct here — rag.tools.web_search owns a lazy
        # singleton and degrades gracefully if TINYFISH_API_KEY is missing.
        pass

    def search(self, state: RAGGraphState) -> RAGGraphState:
        """Performs web search to supplement context."""
        logger.info("Web Search triggered (TinyFish)...")
        question = state["question"]

        web_docs = search_web_sync(question, max_results=3)
        if not web_docs:
            logger.info("Web Search returned no results (or disabled).")
            return state

        existing_docs = state.get("documents", [])
        all_docs = existing_docs + web_docs

        logger.info(f"Web Search found {len(web_docs)} results.")
        return {
            **state,
            "documents": all_docs,
            "web_search_needed": False,  # Handled
        }
