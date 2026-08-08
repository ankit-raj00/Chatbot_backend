"""
Centralized workspace path management.
All subgraphs, tools, and routes import from here.
Never define _DEFAULT_WS inline in individual files.
"""
import os
from pathlib import Path

import structlog
logger = structlog.get_logger(__name__)

# Single source of truth for workspace root
# Set WORKSPACE_ROOT in .env to override
# Default: ~/agentx_workspace (works on Windows, Linux, Mac)
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", str(Path.home() / "agentx_workspace")))

# ── Sandbox UID (Tier 1.1, HARDENING_PLAN.md) ────────────────────────────────
# The backend container runs as root; run_python/run_shell used to execute
# AS that same root user, in the same container, with no OS-level boundary
# at all between "trusted app code" and "arbitrary user-authored code" —
# confirmed exploitable via a live red-team run (unrestricted loopback/
# internal-network reach). Unset (0) by default so local/Windows dev and any
# environment that hasn't opted in are completely unaffected — root creates
# and root runs everything, identical to before this existed. Only the
# Docker image sets SANDBOX_UID, where a dedicated low-privilege user is
# created specifically to run sandboxed subprocesses under.
SANDBOX_UID = int(os.getenv("SANDBOX_UID", "0"))
SANDBOX_GID = int(os.getenv("SANDBOX_GID", str(SANDBOX_UID)))
_SANDBOX_USER_ACTIVE = bool(SANDBOX_UID) and os.name != "nt"


def _chown_for_sandbox(path: Path) -> None:
    """Best-effort (never raises into a caller — a chown failure shouldn't
    break a chat turn), but LOGS on failure rather than silently passing:
    caught a real deploy-config bug this way during testing — os.chown()
    needs CAP_CHOWN specifically (distinct from CAP_SETUID/CAP_SETGID, which
    only cover the subprocess's own identity change, not changing a FILE's
    ownership), and a container capability set that dropped it made every
    sandboxed script unreadable by the UID meant to run it, with the only
    symptom being a generic PermissionError deep in subprocess creation. If
    this is silently swallowed again in some other misconfiguration, the
    sandbox degrades in a way that's hard to diagnose from the outside.

    Non-recursive: hand ownership of a freshly-created (empty) workspace
    directory to the sandbox UID, so a subprocess running AS that UID (not
    root) can actually read/write inside it. Only the directory itself
    needs this — files created later are created BY the sandbox subprocess
    itself and are already owned by it; nothing here needs to be recursive
    except where root itself pre-populates a tree (see venv_python_for,
    which instead just creates the venv AS the sandbox UID to begin with,
    avoiding that case entirely)."""
    if not _SANDBOX_USER_ACTIVE:
        return
    try:
        os.chown(path, SANDBOX_UID, SANDBOX_GID)
    except (PermissionError, AttributeError, OSError) as e:
        logger.warning(
            "sandbox.chown_failed — sandboxed subprocess execution may break",
            path=str(path), sandbox_uid=SANDBOX_UID, error=str(e),
        )


def workspace_for(user_id: str = "anonymous") -> Path:
    """Return (and create) the per-user workspace directory.

    Holds things that are legitimately shared across ALL of a user's
    conversations: the venv, pip/npm caches. User-facing files
    (uploads/outputs/work) live one level deeper, per conversation — see
    conversation_workspace_for().
    """
    ws = WORKSPACE_ROOT / user_id
    ws.mkdir(parents=True, exist_ok=True)
    _chown_for_sandbox(ws)
    return ws


def conversation_workspace_for(user_id: str, conversation_id: str) -> Path:
    """Return (and create) the per-conversation sandbox directory
    (uploads/outputs/work), nested under the user's shared workspace.

    WHY per-conversation and not just per-user: uploads/outputs/work used to
    be a single directory shared by every conversation a user has. The agent
    detects "what file did I just create" by snapshotting outputs/ before a
    turn and diffing mtimes after — with a shared directory, two conversations
    for the same user running turns close together in time could have one
    turn's file write land inside the other turn's snapshot window, silently
    attaching the WRONG conversation's file to a reply. Confirmed live: a
    "transformer PDF" chat and a "Titanic dataset report" chat for the same
    user, ~9s apart, and the Titanic reply came back with the transformer
    chat's PDF attached. Scoping outputs/work/uploads per conversation makes
    that impossible — two conversations now write to physically different
    directories, no locking or serialization needed. The venv/pip cache stay
    shared at the user level (workspace_for) since re-creating those per
    conversation would be pure waste.
    """
    conversations_dir = workspace_for(user_id) / "conversations"
    conversations_dir.mkdir(parents=True, exist_ok=True)
    _chown_for_sandbox(conversations_dir)

    ws = conversations_dir / conversation_id
    ws.mkdir(parents=True, exist_ok=True)
    _chown_for_sandbox(ws)

    for sub in ("uploads", "outputs", "work"):
        sub_dir = ws / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        _chown_for_sandbox(sub_dir)
    return ws


def ensure_workspace():
    """Ensure the root workspace directory exists."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT


def venv_python_for(user_id: str) -> Path:
    """
    Returns the path to the per-user venv's python executable, creating the venv
    on first call.

    NOTE: creating a venv is a slow, BLOCKING subprocess. Do not call this
    directly from an async request path — offload it, e.g.
    `await asyncio.to_thread(venv_python_for, user_id)` — so a first-time venv
    build for one user doesn't stall the whole event loop for every other user.
    """
    import subprocess
    import sys
    ws = workspace_for(user_id)  # already chowned to the sandbox UID if active
    venv_dir = ws / ".venv"
    python_path = (venv_dir / "Scripts" / "python.exe") if os.name == "nt" else (venv_dir / "bin" / "python")
    if not python_path.exists():
        try:
            # Create the venv AS the sandbox UID from the start (parent `ws`
            # is already writable by it) rather than as root then needing a
            # recursive chown of everything `python -m venv` creates — the
            # subprocess that will actually USE this venv (run_python) runs
            # as the same UID, so this keeps ownership consistent everywhere.
            kwargs = {}
            if _SANDBOX_USER_ACTIVE:
                kwargs["user"] = SANDBOX_UID
                kwargs["group"] = SANDBOX_GID
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True, capture_output=True, **kwargs,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create venv for user {user_id}: "
                f"{e.stderr.decode('utf-8', 'replace') if e.stderr else e}"
            ) from e
    return python_path


def pip_cache_dir_for(user_id: str) -> Path:
    d = workspace_for(user_id) / ".cache" / "pip"
    d.mkdir(parents=True, exist_ok=True)
    _chown_for_sandbox(d)
    return d


def npm_prefix_for(user_id: str) -> Path:
    d = workspace_for(user_id) / ".npm-global"
    d.mkdir(parents=True, exist_ok=True)
    _chown_for_sandbox(d)
    return d


def is_path_within_conversation_sandbox(user_id: str, conversation_id: str, path_str: str) -> bool:
    ws_root = conversation_workspace_for(user_id, conversation_id).resolve()
    target = (ws_root / path_str).resolve() if not os.path.isabs(path_str) else Path(path_str).resolve()
    try:
        target.relative_to(ws_root)
        return True
    except ValueError:
        return False
