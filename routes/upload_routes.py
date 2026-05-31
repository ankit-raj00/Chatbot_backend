from core.middleware import get_current_user
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from rag.ingestion_service import IngestionService
import logging
from core.limiter import limiter
import os

# Define Router
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])

# Service Instance (Singleton pattern for simplicity)
ingestion_service = IngestionService()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/upload")
@limiter.limit(f"{os.getenv('UPLOAD_RATE_LIMIT', '5')}/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form("Auto (Detect)"),
    current_user: dict = Depends(get_current_user)
):
    """
    Uploads a file for Agentic RAG Ingestion.
    Automatic Routing, Parsing, Splitting, and Indexing.
    Manual override available via document_type.
    """
    try:
        user_id = str(current_user.get("_id"))
        logger.info(f"Received upload request for: {file.filename} (Type: {document_type}, User: {user_id})")
        
        # Delegate to the Engine
        result = await ingestion_service.process_upload(file, document_type, user_id=user_id)
        
        return {
            "message": "Ingestion successful",
            "details": result
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
