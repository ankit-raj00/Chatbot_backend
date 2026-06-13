"""
Workspace cleanup — deletes generated files older than MAX_AGE_HOURS.
Runs as an asyncio background task inside the FastAPI process.
"""

import asyncio
import os
import time
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

WORKSPACE_ROOT   = Path(os.getenv("WORKSPACE_ROOT", "/tmp/agentx_workspace"))
MAX_AGE_HOURS    = int(os.getenv("WORKSPACE_CLEANUP_HOURS", "24"))
INTERVAL_SECONDS = 3600  # run every hour


async def _cleanup():
    if not WORKSPACE_ROOT.exists():
        return
    cutoff = time.time() - MAX_AGE_HOURS * 3600
    deleted = 0
    for fp in WORKSPACE_ROOT.rglob("*"):
        if fp.is_file() and fp.stat().st_mtime < cutoff:
            try:
                fp.unlink()
                deleted += 1
            except Exception:
                pass
    if deleted:
        logger.info(f"workspace_cleanup.deleted count={deleted}")


async def run_cleanup_loop():
    """Background loop — call asyncio.create_task(run_cleanup_loop()) in lifespan."""
    while True:
        await asyncio.sleep(INTERVAL_SECONDS)
        try:
            await _cleanup()
        except Exception as e:
            logger.error(f"workspace_cleanup.error error={e}")
