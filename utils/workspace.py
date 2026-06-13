"""
Centralized workspace path management.
All subgraphs, tools, and routes import from here.
Never define _DEFAULT_WS inline in individual files.
"""
import os
from pathlib import Path

# Single source of truth for workspace root
# Set WORKSPACE_ROOT in .env to override
# Default: ~/agentx_workspace (works on Windows, Linux, Mac)
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", str(Path.home() / "agentx_workspace")))


def workspace_for(user_id: str = "anonymous") -> Path:
    """Return (and create) the per-user workspace directory."""
    ws = WORKSPACE_ROOT / user_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def ensure_workspace():
    """Ensure the root workspace directory exists."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT
