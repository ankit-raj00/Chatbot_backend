from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from rag.ingestion_service import IngestionService
import logging

# Define Router
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])

# Service Instance (Singleton pattern for simplicity)
ingestion_service = IngestionService()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    document_type: str = Form("Auto (Detect)")
):
    """
    Uploads a file for Agentic RAG Ingestion.
    Automatic Routing, Parsing, Splitting, and Indexing.
    Manual override available via document_type.
    """
    try:
        logger.info(f"Received upload request for: {file.filename} (Type: {document_type})")
        
        # Delegate to the Engine
        result = await ingestion_service.process_upload(file, document_type)
        
        return {
            "message": "Ingestion successful",
            "details": result
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
