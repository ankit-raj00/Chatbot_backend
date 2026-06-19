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


def venv_python_for(user_id: str) -> Path:
    """
    Returns the path to the per-user venv's python executable,
    creating the venv on first call.
    """
    import subprocess
    import sys
    ws = workspace_for(user_id)
    venv_dir = ws / ".venv"
    if not venv_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True, capture_output=True,
        )
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def pip_cache_dir_for(user_id: str) -> Path:
    d = workspace_for(user_id) / ".cache" / "pip"
    d.mkdir(parents=True, exist_ok=True)
    return d


def npm_prefix_for(user_id: str) -> Path:
    d = workspace_for(user_id) / ".npm-global"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_path_within_sandbox(user_id: str, path_str: str) -> bool:
    ws_root = workspace_for(user_id).resolve()
    target = (workspace_for(user_id) / path_str).resolve() if not os.path.isabs(path_str) else Path(path_str).resolve()
    try:
        target.relative_to(ws_root)
        return True
    except ValueError:
        return False
