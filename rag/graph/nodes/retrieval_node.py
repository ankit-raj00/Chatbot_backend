import logging
# from typing import Dict, Any
from rag.graph.state import RAGGraphState
from rag.vector_store.qdrant_manager import QdrantManager
from qdrant_client import models
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalNode:
    """
    The 'Retriever' node responsible for fetching relevant documents.
    Implements Hybrid Search + MMR (Defense #3).
    """
    
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.vector_store = self.qdrant_manager.get_vector_store()
        
    def retrieve(self, state: RAGGraphState) -> RAGGraphState:
        """
        Retrieves documents based on the current question.
        Uses MMR (Maximal Marginal Relevance) to ensure diversity.
        """
        question = state["question"]
        logger.info(f"Retrieving for question: {question}")
        
        # --- Defense #3: MMR Re-ranking ---
        # lambda_mult=0.5 balances Relevance (Vector Similarity) vs Diversity.
        # Fallback to simple similarity search for V1 Verification
        search_strategy = "similarity" # or "mmr"
        logger.info(f"   🔍 Executing Search: [Strategy: {search_strategy}] [k=5]")
        
        # --- Context Filtering (Agentic RAG) ---
        search_kwargs = {"k": 5}
        
        selected_files = state.get("selected_files")
        if selected_files and isinstance(selected_files, list) and len(selected_files) > 0:
            logger.info(f"   🛡️ Applying File Filter: {selected_files}")
            # Construct Qdrant Filter
            # field="metadata.source" matches the key in our Payload
            # match={"any": [...]} allows matching ANY of the selected files
            file_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchAny(any=selected_files)
                    )
                ]
            )
            search_kwargs["filter"] = file_filter
        
        retriever = self.vector_store.as_retriever(
            search_type=search_strategy,
            search_kwargs=search_kwargs
        )
        
        try:
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} documents.")
            
            return {
                "question": question,
                "documents": documents,
                "generation": state.get("generation"),
                "web_search_needed": False, # Reset default
                "hallucination_count": state.get("hallucination_count", 0),
                "retry_count": state.get("retry_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            # Fail gracefully by returning empty list (Router will catch this)
            return {
                **state, 
                "documents": []
            }
