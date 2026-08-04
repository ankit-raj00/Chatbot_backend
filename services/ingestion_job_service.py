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
USER_INDEX_KEY_PREFIX = "ingest_jobs_by_user"
# Bounds the per-user index — old job_ids just age out of the list; the
# individual job records themselves still expire independently via JOB_TTL.
USER_INDEX_MAX_JOBS = 20


class IngestionJobService:

    @staticmethod
    def _key(job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}:{job_id}"

    @staticmethod
    def _user_index_key(user_id: str) -> str:
        return f"{USER_INDEX_KEY_PREFIX}:{user_id}"

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

        # Register in the per-user index so a client that wasn't watching the
        # modal when this job was created (or reloaded the page) can still
        # discover it — see list_jobs_for_user(). Best-effort: losing this
        # index entry only means the job silently doesn't show up in the
        # persistent status view, it doesn't affect the job itself.
        try:
            index = await cache_get(cls._user_index_key(user_id)) or []
            index = [job_id] + [j for j in index if j != job_id]
            index = index[:USER_INDEX_MAX_JOBS]
            await cache_set(cls._user_index_key(user_id), index, ttl_seconds=JOB_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to update job index for user {user_id}: {e}")

        return job_id

    @classmethod
    async def list_jobs_for_user(cls, user_id: str) -> list[dict]:
        """Every job in this user's index that hasn't expired yet, newest
        first. Used for a persistent status view that survives the upload
        modal closing or the page reloading — unlike get_job(), which
        requires already knowing a specific job_id."""
        index = await cache_get(cls._user_index_key(user_id)) or []
        jobs = []
        for job_id in index:
            job = await cls.get_job(job_id)
            if job is not None:
                jobs.append(job)
        return jobs

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
