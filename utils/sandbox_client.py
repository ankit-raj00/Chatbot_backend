"""
Client for SES (the Sandbox Execution Service) — Tier 2.1, see
../../TIER2_SANDBOX_PLAN.md.

When SANDBOX_EXECUTOR_MODE=remote, run_python/run_shell execute inside gVisor
containers on a dedicated sandbox host instead of as subprocesses in this
container. Every function here yields the SAME item shape as
utils/code_executor.py's stream_python/stream_shell —
    {"line": str, "stream": "stdout"|"stderr"}  …then…  {"done": True, "exit_code": int}
— so the tool call sites barely change.

FAIL-CLOSED BY DESIGN: if SES is unreachable in remote mode, the tool call
returns an error. It must NEVER silently fall back to local execution — that
would quietly drop the security boundary at exactly the moment something is
wrong. The mode flag is the only way back.
"""
import json
import os
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

SANDBOX_EXECUTOR_MODE = os.getenv("SANDBOX_EXECUTOR_MODE", "local").strip().lower()
SANDBOX_SERVICE_URL = os.getenv("SANDBOX_SERVICE_URL", "").rstrip("/")
SANDBOX_SERVICE_TOKEN = os.getenv("SANDBOX_SERVICE_TOKEN", "")
# Connect timeout is short (a dead host should fail fast); read timeout is None
# because an execution legitimately streams for minutes — SES enforces the real
# deadline itself and kills the container.
_CONNECT_TIMEOUT = float(os.getenv("SANDBOX_CONNECT_TIMEOUT", "10"))


def is_remote() -> bool:
    """True only if remote mode is fully configured — a half-configured remote
    mode silently degrading to local is exactly what fail-closed forbids, so a
    missing URL/token is surfaced loudly at call time instead."""
    return SANDBOX_EXECUTOR_MODE == "remote"


def _headers() -> dict:
    return {"Authorization": f"Bearer {SANDBOX_SERVICE_TOKEN}",
            "Content-Type": "application/json"}


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=_CONNECT_TIMEOUT, read=None, write=30.0, pool=_CONNECT_TIMEOUT)


async def _stream_ndjson(path: str, payload: dict):
    """POST and yield parsed NDJSON items as they arrive."""
    if not SANDBOX_SERVICE_URL or not SANDBOX_SERVICE_TOKEN:
        yield {"line": "SANDBOX UNAVAILABLE: remote execution is enabled but "
                       "SANDBOX_SERVICE_URL/SANDBOX_SERVICE_TOKEN are not configured.",
               "stream": "stderr"}
        yield {"done": True, "exit_code": -1}
        return

    url = f"{SANDBOX_SERVICE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=payload, headers=_headers()) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    logger.error("sandbox.http_error", url=url,
                                 status=resp.status_code, body=body)
                    yield {"line": f"SANDBOX ERROR: execution service returned "
                                   f"HTTP {resp.status_code}. {body}",
                           "stream": "stderr"}
                    yield {"done": True, "exit_code": -1}
                    return
                async for raw in resp.aiter_lines():
                    if not raw.strip():
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("sandbox.bad_ndjson_line", line=raw[:200])
    except httpx.HTTPError as e:
        # Fail closed — never fall through to local execution here.
        logger.error("sandbox.unreachable", url=url, error=str(e))
        yield {"line": "SANDBOX UNAVAILABLE: could not reach the code execution "
                       "service, so this code was not run. Nothing was executed. "
                       "Explain this to the user rather than retrying repeatedly.",
               "stream": "stderr"}
        yield {"done": True, "exit_code": -1}


async def stream_python_remote(user_id: str, conversation_id: str, code: str,
                               filename: str | None = None, timeout: int = 300):
    async for item in _stream_ndjson("/exec/python", {
        "user_id": user_id, "conversation_id": conversation_id,
        "code": code, "filename": filename, "timeout": timeout,
    }):
        yield item


async def stream_shell_remote(user_id: str, conversation_id: str, command: str,
                              timeout: int = 120):
    async for item in _stream_ndjson("/exec/shell", {
        "user_id": user_id, "conversation_id": conversation_id,
        "command": command, "timeout": timeout,
    }):
        yield item


async def pip_install_remote(user_id: str, conversation_id: str, package: str,
                             timeout: int = 180):
    async for item in _stream_ndjson("/exec/pip-install", {
        "user_id": user_id, "conversation_id": conversation_id,
        "package": package, "timeout": timeout,
    }):
        yield item


async def push_file(user_id: str, conversation_id: str, rel_path: str,
                    data: bytes) -> bool:
    """Push a local file (e.g. a user upload) into the sandbox workspace so the
    agent's code can read it. Best-effort: logged, never raises into a turn."""
    if not SANDBOX_SERVICE_URL:
        return False
    url = f"{SANDBOX_SERVICE_URL}/fs/{user_id}/{conversation_id}/{rel_path}"
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.put(
                url, content=data,
                headers={"Authorization": f"Bearer {SANDBOX_SERVICE_TOKEN}"})
            return resp.status_code == 200
    except httpx.HTTPError as e:
        logger.error("sandbox.push_failed", path=rel_path, error=str(e))
        return False


async def sync_outputs(user_id: str, conversation_id: str, local_outputs: Path) -> list[str]:
    """Pull files generated on the sandbox host into the LOCAL outputs/ dir.

    This is what lets ChatService's outputs snapshot-diff and output_routes.py
    keep working completely unchanged — by the time the turn inspects the local
    directory, the generated files are really there.

    Only fetches what's new or changed (by size+mtime), so a conversation with
    many existing outputs doesn't re-download them every turn.
    """
    if not SANDBOX_SERVICE_URL:
        return []
    base = f"{SANDBOX_SERVICE_URL}/fs/{user_id}/{conversation_id}"
    fetched: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.get(f"{base}/manifest",
                                    headers={"Authorization": f"Bearer {SANDBOX_SERVICE_TOKEN}"})
            if resp.status_code != 200:
                logger.error("sandbox.manifest_failed", status=resp.status_code)
                return []
            manifest = resp.json().get("outputs", [])

            local_outputs.mkdir(parents=True, exist_ok=True)
            for entry in manifest:
                name = entry.get("name")
                if not name or "/" in name or "\\" in name:
                    continue                      # manifest is data, not trusted input
                target = local_outputs / name
                if target.exists():
                    st = target.stat()
                    if st.st_size == entry.get("size") and st.st_mtime >= entry.get("mtime", 0):
                        continue                  # already have this exact version
                file_resp = await client.get(
                    f"{base}/outputs/{name}",
                    headers={"Authorization": f"Bearer {SANDBOX_SERVICE_TOKEN}"})
                if file_resp.status_code == 200:
                    target.write_bytes(file_resp.content)
                    fetched.append(name)
    except httpx.HTTPError as e:
        logger.error("sandbox.sync_outputs_failed", error=str(e))
    if fetched:
        logger.info("sandbox.outputs_synced", count=len(fetched), files=fetched)
    return fetched
