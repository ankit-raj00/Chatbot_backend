"""
Workspace cleanup — deletes generated files and idle virtual environments based on
a hierarchical directory policy. Runs as an asyncio background task.
"""

import asyncio
import json
import os
import time
from pathlib import Path
import structlog
from utils.workspace import WORKSPACE_ROOT

logger = structlog.get_logger(__name__)

INTERVAL_SECONDS = 3600  # run every hour

# (directory_name, max_age_hours, "files" | "tree")
# NOTE: the local workspace is a CACHE, not the source of truth — generated
# outputs and uploads are also persisted to Cloudinary, so eviction here only
# drops the fast local copy (re-hydrated on demand from Cloudinary when served).
# "files" mode is now also gated on workspace idleness (see _cleanup), so an
# actively-used workspace keeps its files locally regardless of file age.
CLEANUP_POLICY = [
    ("outputs",  168,  "files"),  # generated deliverables — keep 7d locally (durable copy in Cloudinary)
    ("uploads",  168,  "files"),  # user uploads — keep 7d locally (durable copy in Cloudinary)
    ("work",     48,   "files"),  # scratch/intermediate files
    (".venv",    336,  "tree"),   # venv — wipe after 14 DAYS of inactivity
    (".npm-global", 336, "tree"), # npm prefix — same
    (".cache",   336,  "tree"),   # pip/npm cache — 14 days
]

# Only reap "files"-mode dirs once the whole workspace has been idle this long.
# Prevents deleting an active user's recent artifacts purely on file mtime.
FILES_IDLE_GRACE_HOURS = 24


def _read_last_active(user_dir: Path) -> float:
    meta = user_dir / ".meta" / "last_active.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("timestamp", 0)
        except Exception:
            return 0
    return 0


def touch_last_active(user_id: str, project_type: str = "") -> None:
    """Call this from agent_tool_node after EVERY run_python/run_shell call."""
    from utils.workspace import workspace_for
    meta_dir = workspace_for(user_id) / ".meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "last_active.json").write_text(json.dumps({
        "timestamp": time.time(),
        "project_type": project_type,
    }))


async def _cleanup():
    if not WORKSPACE_ROOT.exists():
        return
    now = time.time()
    for user_dir in WORKSPACE_ROOT.iterdir():
        if not user_dir.is_dir():
            continue
        last_active = _read_last_active(user_dir)
        idle_hours = (now - last_active) / 3600 if last_active else 999999

        # "files" targets: uploads/outputs/work now live per-conversation
        # (see utils.workspace.conversation_workspace_for), nested under
        # <user>/conversations/<conversation_id>/. Old flat <user>/outputs/
        # etc. can still exist for files created before that migration, so
        # both layouts are swept. "tree" targets (.venv, caches) stay at the
        # user root — those are still legitimately shared across conversations.
        file_targets = []
        for dirname, max_age_h, mode in CLEANUP_POLICY:
            if mode != "files":
                continue
            legacy = user_dir / dirname
            if legacy.exists():
                file_targets.append((legacy, max_age_h))
            conversations_root = user_dir / "conversations"
            if conversations_root.exists():
                for conv_dir in conversations_root.iterdir():
                    target = conv_dir / dirname
                    if target.exists():
                        file_targets.append((target, max_age_h))

        # Gate on workspace idleness: don't evict files from a workspace
        # that's still in active use, even if individual files are old.
        if idle_hours >= FILES_IDLE_GRACE_HOURS:
            for target, max_age_h in file_targets:
                cutoff = now - max_age_h * 3600
                for fp in target.rglob("*"):
                    if fp.is_file() and fp.stat().st_mtime < cutoff:
                        try:
                            fp.unlink()
                        except Exception:
                            pass

        for dirname, max_age_h, mode in CLEANUP_POLICY:
            if mode != "tree":
                continue
            target = user_dir / dirname
            if not target.exists():
                continue
            if idle_hours >= max_age_h:
                try:
                    import shutil as _sh
                    _sh.rmtree(target)
                    logger.info(f"workspace_cleanup.tree_removed user={user_dir.name} dir={dirname} idle_h={idle_hours:.1f}")
                except Exception as e:
                    logger.warning(f"workspace_cleanup.tree_remove_failed dir={target} error={e}")


async def run_cleanup_loop():
    while True:
        await asyncio.sleep(INTERVAL_SECONDS)
        try:
            await _cleanup()
        except Exception as e:
            logger.error(f"workspace_cleanup.error error={e}")
