from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from rag.graph.workflow import RAGWorkflow
import logging
from typing import List, Optional
from rag.tools.doc_store_tools import read_document_page
from core.middleware import get_current_user

# Define Router
router = APIRouter(prefix="/api/v1/rag", tags=["Agentic RAG"])

# Initialize Workflow (Singleton for compilation efficientcy)
workflow = RAGWorkflow()
app = workflow.get_app()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    selected_file_ids: Optional[List[str]] = None  # list of UUID file_ids, not filenames

class PageReadRequest(BaseModel):
    doc_id: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    hallucination_warning: bool

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Agentic RAG Chat Endpoint.
    Runs the full Retrieve-Grade-Generate-Validate loop.
    """
    try:
        user_id = str(current_user.get("_id"))
        logger.info(f"Received RAG query: {request.message} (user={user_id})")
        if request.selected_file_ids:
            logger.info(f"Context Filters detected: {len(request.selected_file_ids)} files")


        # Initial State — user_id is the mandatory tenancy filter (see
        # RAGGraphState); this endpoint used to omit it entirely, so
        # retrieval always came back empty (auth-less request, no user to
        # filter by) and the agent fell back to calling search_knowledge_base
        # itself with no user context, self-rejecting via its tenancy guard.
        inputs = {
            "question": request.message,
            "retry_count": 0,
            "hallucination_count": 0,
            "selected_file_ids": request.selected_file_ids,  # UUIDs for Qdrant filter
            "user_id": user_id,
        }
        
        # Invoke Graph
        # We use invoke() for synchronous waiting. For async streaming we'd use astream.
        final_state = await app.ainvoke(inputs)
        
        # Extract Results
        answer = final_state.get("generation", "No answer generated.")
        if isinstance(answer, list):
            # Extract text from list of content blocks if necessary
            texts = []
            for block in answer:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            answer = "\n".join(texts)
            
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
async def retrieve_only(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    DEBUG ENDPOINT: Retrieve only.
    Bypasses Grader/Generator to show raw chunks found by vector search.
    Auth-required and user-scoped — retrieval only returns the caller's own docs.
    """
    try:
        user_id = str(current_user.get("_id"))
        logger.info(f"Retrieving chunks for: {request.message} (user={user_id})")

        retriever_node = workflow.retriever
        # user_id is the mandatory tenancy filter enforced inside retrieve()
        state = {"question": request.message, "user_id": user_id}
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
async def list_files(current_user: dict = Depends(get_current_user)):
    """
    Lists unique source files currently in the Vector DB for the current user.
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
        user_id = str(current_user.get("_id"))
        sources = manager.list_unique_sources(user_id=user_id)
        return {"files": sources}
    except Exception as e:
        logger.error(f"List files failed: {str(e)}")
        return {"files": []}

@router.delete("/file/{file_id}")
async def delete_file(file_id: str, current_user: dict = Depends(get_current_user)):
    """
    Deletes a file and all its associated chunks from the Vector DB.
    """
    try:
        from rag.vector_store.qdrant_manager import QdrantManager
        manager = QdrantManager()
        user_id = str(current_user.get("_id"))
        
        success = manager.delete_file(file_id=file_id, user_id=user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete file from Qdrant")
            
        return {"status": "success", "message": f"File {file_id} deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
