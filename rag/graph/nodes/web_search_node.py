import logging
import os
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from rag.graph.state import RAGGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchNode:
    """
    The 'Web Search' node for fallback retrieval.
    Implements Defense #4: Hybrid Search expansion (Internal + External).
    """
    
    def __init__(self):
        try:
            self.tool = TavilySearch(max_results=3)
        except Exception as e:
            logger.warning(f"Tavily Search not initialized (API Key missing?): {e}")
            self.tool = None

    def search(self, state: RAGGraphState) -> RAGGraphState:
        """
        Performs web search to supplement context.
        """
        logger.info("Web Search triggered...")
        question = state["question"]
        
        if not self.tool:
            logger.error("Web Search skipped: Tool not initialized.")
            return state

        try:
            # Tavily returns list of dicts: [{'content': '...', 'url': '...'}]
            results = self.tool.invoke({"query": question})
            
            web_docs = []
            if isinstance(results, list):
                for res in results:
                    web_docs.append(
                        Document(
                            page_content=res.get("content", ""), 
                            metadata={"source": res.get("url", "web_search")}
                        )
                    )
            
            # Combine with existing docs (if any, though usually this node is hit when docs are empty)
            existing_docs = state.get("documents", [])
            all_docs = existing_docs + web_docs
            
            logger.info(f"Web Search found {len(web_docs)} results.")
            
            return {
                **state,
                "documents": all_docs,
                "web_search_needed": False # Handled
            }
            
        except Exception as e:
            logger.error(f"Web Search failed: {str(e)}")
            return state
