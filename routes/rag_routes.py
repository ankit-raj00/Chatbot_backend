from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from rag.graph.workflow import RAGWorkflow
import logging
from typing import List, Optional
from rag.tools.doc_store_tools import read_document_page

# Define Router
router = APIRouter(prefix="/api/v1/rag", tags=["Agentic RAG"])

# Initialize Workflow (Singleton for compilation efficientcy)
workflow = RAGWorkflow()
app = workflow.get_app()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    selected_files: Optional[List[str]] = None

class PageReadRequest(BaseModel):
    doc_id: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    hallucination_warning: bool

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Agentic RAG Chat Endpoint.
    Runs the full Retrieve-Grade-Generate-Validate loop.
    """
    try:
        logger.info(f"Received RAG query: {request.message}")
        if request.selected_files:
            logger.info(f"Context Filters detected: {len(request.selected_files)} files")
        
        # Initial State
        inputs = {
            "question": request.message,
            "retry_count": 0,
            "hallucination_count": 0,
            "selected_files": request.selected_files # Pass filters to the graph
        }
        
        # Invoke Graph
        # We use invoke() for synchronous waiting. For async streaming we'd use astream.
        final_state = await app.ainvoke(inputs)
        
        # Extract Results
        answer = final_state.get("generation", "No answer generated.")
        documents = final_state.get("documents", [])
        
        # Extract Sources (Metadata)
        sources = [doc.metadata.get("source", "unknown") for doc in documents]
        # Deduplicate sources
        sources = list(set(sources))
        
        # Check termination reason
        hallucination_warning = final_state.get("hallucination_count", 0) > 0
        
        return {
            "answer": answer,
            "sources": sources,
            "hallucination_warning": hallucination_warning
        }
        
    except Exception as e:
        logger.error(f"RAG Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
async def retrieve_only(request: ChatRequest):
    """
    DEBUG ENDPOINT: Retrieve only.
    Bypasses Grader/Generator to show raw chunks found by vector search.
    """
    try:
        logger.info(f"Retrieving chunks for: {request.message}")
        inputs = {"question": request.message}
        
        # Manually run just the retrieval node logic
        # (Or compile a sub-graph, but manual node call is easier since it's just a method)
        # We access the retriever instance from the workflow object
        retriever_node = workflow.retriever
        
        # Need to construct a minimal state
        state = {"question": request.message}
        result_state = retriever_node.retrieve(state)
        
        documents = result_state.get("documents", [])
        
        # Format for frontend
        chunks = []
        for doc in documents:
            chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0) # Qdrant wrapper might put score in metadata or separate
            })
            
        return {"chunks": chunks, "count": len(chunks)}
        
    except Exception as e:
        logger.error(f"Retrieval check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_files():
    """
    Lists unique source files currently in the Vector DB.
    """
    try:
        # We need access to QdrantManager. 
        # It's instantiated inside IngestionService AND RetrievalNode (via retriever wrapper).
        # Let's use the one in RetrievalNode for consistency.
        # wrapper -> vectorstore -> client... actually QdrantManager is a wrapper around the client.
        # RetrievalNode initializes `self.vectorstore = QdrantManager().get_vector_store()`
        # We need the MANAGER instance to call `list_unique_sources`.
        # Since QdrantManager is a Singleton now, we can just instantiate it.
        from rag.vector_store.qdrant_manager import QdrantManager
        manager = QdrantManager()
        sources = manager.list_unique_sources()
        return {"files": sources}
    except Exception as e:
        logger.error(f"List files failed: {str(e)}")
        # Return empty list on error to not break frontend
        return {"files": []}

@router.post("/read-page")
async def read_page_tool(request: PageReadRequest):
    """
    TOOL ENDPOINT: Reads a specific page from the MongoDB DocStore.
    Used by Agents (or manual testing) to deep-dive into content.
    """
    try:
        content = await read_document_page.ainvoke({"doc_id": request.doc_id, "page_number": request.page})
        if not content:
            raise HTTPException(status_code=404, detail="Page not found or invalid DocID")
        return content
    except Exception as e:
        logger.error(f"Read page tool failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
