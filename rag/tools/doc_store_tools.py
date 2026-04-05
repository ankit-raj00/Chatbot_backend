from core.database import doc_store_collection
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

from langchain_core.tools import tool

@tool
async def read_document_page(doc_id: str, page_number: int):
    """
    Retrieves the full content of a specific page from the MongoDB DocStore.
    
    IMPORTANT: You must use the 'json_id' from the chunk's metadata as the 'doc_id'.
    
    Args:
        doc_id (str): The 'json_id' found in the metadata of the retrieved chunks.
        page_number (int): The page number to retrieve (1-indexed).
        
    Returns:
        dict: The content of the page (markdown, text, images), or None if not found.
    """
    try:
        # 1. Validate ObjectId
        if not ObjectId.is_valid(doc_id):
            logger.error(f"Invalid DocStore ID: {doc_id}")
            return None

        # 2. Query MongoDB
        query = {
            "_id": ObjectId(doc_id),
            "pages": { "$elemMatch": { "page": page_number } }
        }
        
        # Projection: We only want the matching page from the pages array
        projection = {
            "pages.$": 1, 
            "source": 1,
            "document_id": 1 
        }
        
        result = await doc_store_collection.find_one(query, projection)
        
        if not result or "pages" not in result or not result["pages"]:
            logger.warning(f"Page {page_number} not found in DocStore {doc_id}")
            return None
            
        # 3. Extract Page Data
        page_data = result["pages"][0] # $elemMatch returns the single matching element
        
        # Add source info for context
        page_data["source"] = result.get("source")
        page_data["doc_store_id"] = str(result["_id"])
        
        return page_data

    except Exception as e:
        logger.error(f"Error reading page content: {e}")
        return None
