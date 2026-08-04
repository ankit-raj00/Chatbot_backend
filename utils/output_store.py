"""
Output cache <-> Cloudinary durability layer.

The per-user local workspace `outputs/` folder is a CACHE. The durable source of
truth for every generated file is Cloudinary (URL) + MongoDB `user_outputs` (the
index of what exists). So a file missing locally (evicted by cleanup, or gone
after a server restart/redeploy) is always recoverable from Cloudinary.

Restore is LAZY and NON-BLOCKING:
  - Serving (user preview/download) re-hydrates a single file on demand (see
    routes/output_routes.py -> _serve_cloudinary).
  - For the AGENT (which reads the raw filesystem via run_python/run_shell),
    `restore_outputs_background` refills all missing files in parallel, kicked off
    as a background task so it never blocks the first streamed token.
"""
import asyncio
import structlog

from utils.workspace import conversation_workspace_for
from utils.cloudinary_handler import CloudinaryHandler

logger = structlog.get_logger(__name__)


async def ensure_output_local(user_id: str, conversation_id: str, filename: str) -> bool:
    """Ensure a single tracked output exists locally, fetching from Cloudinary if
    missing. Returns True if the file is present locally afterwards."""
    from core.database import user_outputs_collection

    outputs_dir = conversation_workspace_for(user_id, conversation_id) / "outputs"
    local_path = outputs_dir / filename
    if local_path.exists():
        return True

    doc = await user_outputs_collection.find_one(
        {"user_id": user_id, "conversation_id": conversation_id, "filename": filename}, {"cloudinary_url": 1}
    )
    url = doc.get("cloudinary_url") if doc else None
    if not url:
        return False
    try:
        await CloudinaryHandler().download_file(url, target_path=str(local_path))
        return local_path.exists()
    except Exception as e:
        logger.warning("output_restore.failed", filename=filename, error=str(e))
        return False


async def restore_outputs_background(user_id: str, conversation_id: str) -> None:
    """Restore ALL missing tracked outputs for THIS conversation, in PARALLEL.

    Non-blocking by contract: schedule via utils.background_tasks.spawn so the
    chat response is never delayed. Files already on disk are skipped.
    """
    from core.database import user_outputs_collection

    outputs_dir = conversation_workspace_for(user_id, conversation_id) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    handler = CloudinaryHandler()

    async def _one(filename: str, url: str) -> None:
        if not filename or not url:
            return
        local_path = outputs_dir / filename
        if local_path.exists():
            return
        try:
            await handler.download_file(url, target_path=str(local_path))
        except Exception as e:
            logger.warning("output_restore.failed", filename=filename, error=str(e))

    cursor = user_outputs_collection.find(
        {"user_id": user_id, "conversation_id": conversation_id}, {"filename": 1, "cloudinary_url": 1}
    )
    tasks = [
        _one(doc.get("filename"), doc.get("cloudinary_url"))
        async for doc in cursor
    ]
    if tasks:
        await asyncio.gather(*tasks)
        logger.info("output_restore.done", user_id=user_id, conversation_id=conversation_id, count=len(tasks))
