"""
IngestionJobService — manages background ingestion job lifecycle.
Uses Redis to store job status with 24-hour TTL.

Job status schema:
{
    "job_id": "uuid",
    "status": "queued" | "parsing" | "embedding" | "complete" | "failed",
    "filename": "document.pdf",
    "user_id": "...",
    "progress_message": "Parsing with LlamaParse...",
    "file_id": "uuid (set when complete)",
    "chunks_count": 42 (set when complete),
    "error": "error message (set when failed)",
    "created_at": "ISO timestamp",
    "updated_at": "ISO timestamp"
}
"""

import uuid
import logging
from datetime import datetime, timezone

from core.cache import cache_set, cache_get

import structlog
logger = structlog.get_logger(__name__)

JOB_TTL_SECONDS = 86400    # 24 hours
JOB_KEY_PREFIX = "ingest_job"


class IngestionJobService:

    @staticmethod
    def _key(job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}:{job_id}"

    @classmethod
    async def create_job(cls, filename: str, user_id: str) -> str:
        """Create a new job entry in Redis. Returns the job_id."""
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "user_id": user_id,
            "progress_message": "Queued for processing",
            "file_id": None,
            "chunks_count": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await cache_set(cls._key(job_id), job_data, ttl_seconds=JOB_TTL_SECONDS)
        return job_id

    @classmethod
    async def update_job(cls, job_id: str, **fields) -> None:
        """Update specific fields on an existing job."""
        existing = await cache_get(cls._key(job_id))
        if existing is None:
            logger.warning(f"Job {job_id} not found in Redis during update")
            return
        existing.update(fields)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        await cache_set(cls._key(job_id), existing, ttl_seconds=JOB_TTL_SECONDS)

    @classmethod
    async def get_job(cls, job_id: str) -> dict | None:
        """Retrieve job status. Returns None if job not found or expired."""
        return await cache_get(cls._key(job_id))
