"""
Upload routes — non-blocking ingestion via background tasks.

Flow:
  POST /upload → validates input, saves temp file, returns job_id immediately
  Background task runs the full pipeline asynchronously
  GET  /job/{job_id} → client polls this for status
"""

import os
import uuid
import shutil
import logging
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from core.middleware import get_current_user
from core.limiter import limiter
from rag.ingestion_service import IngestionService
from services.ingestion_job_service import IngestionJobService

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)

# Supported file types
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx", ".csv", ".json", ".py", ".js", ".ts"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB


async def _run_ingestion_background(
    job_id: str,
    temp_path: str,
    original_filename: str,
    document_type: str,
    user_id: str,
) -> None:
    """
    Background task: runs the full ingestion pipeline.
    Updates Redis job status at each stage.
    Cleans up the temp file when done (success or failure).
    """
    try:
        await IngestionJobService.update_job(
            job_id,
            status="parsing",
            progress_message="Parsing document with LlamaParse..."
        )

        # Re-create an UploadFile-like object from the saved temp file
        # IngestionService expects an UploadFile, but we already saved the file
        # We pass the temp path directly to the underlying processor
        ingestion_service = IngestionService()

        result = await ingestion_service.process_upload_from_path(
            file_path=temp_path,
            filename=original_filename,
            document_type=document_type,
            user_id=user_id,
        )

        await IngestionJobService.update_job(
            job_id,
            status="complete",
            progress_message="Ingestion complete",
            file_id=result.get("file_id"),
            chunks_count=result.get("chunks_count", 0),
        )
        logger.info(f"Ingestion job {job_id} complete: {original_filename}")

    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}", exc_info=True)
        await IngestionJobService.update_job(
            job_id,
            status="failed",
            progress_message="Ingestion failed",
            error=str(e),
        )
    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_err}")


@router.post("/upload")
@limiter.limit(f"{os.getenv('UPLOAD_RATE_LIMIT', '5')}/minute")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = Form("Auto (Detect)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Accepts a file for RAG ingestion.
    Returns a job_id immediately — does NOT wait for ingestion to complete.
    Poll GET /api/v1/ingest/job/{job_id} for status.
    """
    user_id = str(current_user.get("_id"))

    # ── Input validation ─────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # ── Save to temp file ────────────────────────────────────
    # We must read the file content NOW (during the HTTP request)
    # because the UploadFile object is closed after the request ends.
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", f"upload_{uuid.uuid4()}{ext}")

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB"
            )
        with open(temp_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

    # ── Create job record in Redis ───────────────────────────
    job_id = await IngestionJobService.create_job(
        filename=file.filename,
        user_id=user_id,
    )

    # ── Schedule background processing ──────────────────────
    background_tasks.add_task(
        _run_ingestion_background,
        job_id=job_id,
        temp_path=temp_path,
        original_filename=file.filename,
        document_type=document_type,
        user_id=user_id,
    )

    logger.info(f"Ingestion job {job_id} queued for {file.filename} (user: {user_id})")

    return {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "message": "Processing started. Poll /api/v1/ingest/job/{job_id} for status."
    }


@router.get("/job/{job_id}")
async def get_ingestion_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Poll this endpoint to check ingestion progress.
    Returns the current job status from Redis.
    """
    job = await IngestionJobService.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found. It may have expired (jobs are kept for 24 hours)."
        )

    # Security: only the job owner can check status
    if job.get("user_id") != str(current_user.get("_id")):
        raise HTTPException(status_code=403, detail="Access denied")

    return job
