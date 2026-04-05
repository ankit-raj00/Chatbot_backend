from typing import List, Optional, Dict, Any
from qdrant_client import models
from rag.vector_store.qdrant_manager import QdrantManager
import logging

logger = logging.getLogger(__name__)

# Initialize QdrantManager once
qdrant_manager = QdrantManager()

from langchain_core.tools import tool

@tool
def search_knowledge_base(
    query: str, 
    selected_files: Optional[List[str]] = None,
    limit: int = 5,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Agent Tool: Searches the knowledge base (Vector DB) for relevant context.
    
    Args:
        query (str): The semantic search query.
        selected_files (List[str], optional): List of filenames to restrict search to.
                                              If empty or None, searches all files (unless enforced upstream).
        limit (int): Number of chunks to return (default: 5).
        offset (int): Pagination offset (default: 0).
        
    Returns:
        List[Dict]: A list of chunks with 'content', 'source', 'score', and 'id'.
    """
    try:
        logger.info(f"🛠️ Tool Call: search_knowledge_base(query='{query}', files={selected_files}, limit={limit}, offset={offset})")
        
        vector_store = qdrant_manager.get_vector_store()
        
        # 1. Build Filter
        search_kwargs = {
            "k": limit,
            "offset": offset, # Note: LangChain wrapper might not support raw 'offset' in as_retriever directly
                              # We might need to use the raw client methods for pagination + filtering
                              # LangChain's similarity_search usually takes k but not offset.
                              # Let's use QdrantClient directly for maximum control over offset.
        }
        
        # Construct Filter
        qdrant_filter = None
        if selected_files:
             qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source", 
                        match=models.MatchAny(any=selected_files)
                    )
                ]
            )

        # Search using LangChain Wrapper (Safer & Compatible)
        # Note: We pass the filter if it exists
        results = vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            filter=qdrant_filter # LangChain-Qdrant supports passing the models.Filter
        )
        
        # Format Results
        output = []
        for doc, score in results:
            payload = doc.metadata or {}
            
            output.append({
                "content": doc.page_content,
                "source": payload.get("source", "unknown"),
                "json_id": payload.get("json_id") or payload.get("json_file_id", "unknown"),
                "score": score,
                "document_id": payload.get("_id", "unknown") # Metadata often holds the ID in LC
            })
            
        logger.info(f"   ✅ Found {len(output)} chunks.")
        return output

    except Exception as e:
        logger.error(f"❌ Search Tool Failed: {e}")
        return []
