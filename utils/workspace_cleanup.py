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
CLEANUP_POLICY = [
    ("outputs",  6,   "files"),   # generated deliverables — short TTL
    ("uploads",  72,  "files"),   # user uploads — longer grace period
    ("work",     24,  "files"),   # scratch/intermediate files
    (".venv",    168, "tree"),    # venv — wipe after 7 DAYS of inactivity
    (".npm-global", 168, "tree"), # npm prefix — same
    (".cache",   336, "tree"),    # pip/npm cache — 14 days
]


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

        for dirname, max_age_h, mode in CLEANUP_POLICY:
            target = user_dir / dirname
            if not target.exists():
                continue

            if mode == "tree":
                if idle_hours >= max_age_h:
                    try:
                        import shutil as _sh
                        _sh.rmtree(target)
                        logger.info(f"workspace_cleanup.tree_removed user={user_dir.name} dir={dirname} idle_h={idle_hours:.1f}")
                    except Exception as e:
                        logger.warning(f"workspace_cleanup.tree_remove_failed dir={target} error={e}")
            else:  # "files"
                cutoff = now - max_age_h * 3600
                for fp in target.rglob("*"):
                    if fp.is_file() and fp.stat().st_mtime < cutoff:
                        try:
                            fp.unlink()
                        except Exception:
                            pass


async def run_cleanup_loop():
    while True:
        await asyncio.sleep(INTERVAL_SECONDS)
        try:
            await _cleanup()
        except Exception as e:
            logger.error(f"workspace_cleanup.error error={e}")
