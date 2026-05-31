"""
Hook system for agent tool lifecycle events.
Inspired by Claude Agent SDK's lifecycle hooks.

Usage — register a hook:
    from utils.hooks import register_pre_tool_hook

    @register_pre_tool_hook
    async def my_guard(tool_name: str, tool_args: dict, user_id: str) -> dict | None:
        if tool_name == "bash" and "rm -rf" in tool_args.get("command", ""):
            return {"deny": True, "reason": "Destructive bash command blocked by security hook"}
        return None   # None means allow

Usage — in tool node:
    from utils.hooks import run_pre_tool_hooks, run_post_tool_hooks
    result = await run_pre_tool_hooks(tool_name, tool_args, user_id)
    if result and result.get("deny"):
        # return error ToolMessage

Hook return values:
    None                           → allow, continue
    {"deny": True, "reason": "..."} → block this tool call
    {"modify": True, "args": {...}} → replace tool_args with the new args
"""

import time
import logging
from typing import Callable, Awaitable, Any

logger = logging.getLogger(__name__)

# Registered hooks — populated via decorators
_pre_tool_hooks: list[Callable] = []
_post_tool_hooks: list[Callable] = []


# ─── Decorator-based registration ────────────────────────────────────────────

def register_pre_tool_hook(fn: Callable) -> Callable:
    """Decorator to register a pre-tool hook. Must be an async function."""
    _pre_tool_hooks.append(fn)
    logger.debug(f"Registered pre-tool hook: {fn.__name__}")
    return fn


def register_post_tool_hook(fn: Callable) -> Callable:
    """Decorator to register a post-tool hook. Must be an async function."""
    _post_tool_hooks.append(fn)
    logger.debug(f"Registered post-tool hook: {fn.__name__}")
    return fn


# ─── Built-in hooks ──────────────────────────────────────────────────────────

@register_pre_tool_hook
async def _log_tool_call(tool_name: str, tool_args: dict, user_id: str) -> None:
    """Log every tool call with user context."""
    logger.info(
        "tool.pre_call",
        tool_name=tool_name,
        user_id=user_id,
        args_keys=list(tool_args.keys()) if isinstance(tool_args, dict) else [],
    )
    return None   # Allow


@register_post_tool_hook
async def _log_tool_result(tool_name: str, result: Any, duration_ms: float, user_id: str) -> None:
    """Log every tool result with timing."""
    result_preview = str(result)[:100] if result else "None"
    logger.info(
        "tool.post_call",
        tool_name=tool_name,
        user_id=user_id,
        duration_ms=round(duration_ms, 2),
        result_preview=result_preview,
    )


# ─── Runner functions ─────────────────────────────────────────────────────────

async def run_pre_tool_hooks(
    tool_name: str,
    tool_args: dict,
    user_id: str = ""
) -> dict | None:
    """
    Run all registered pre-tool hooks in order.
    Returns the first non-None result (deny or modify).
    Returns None if all hooks pass.
    """
    for hook in _pre_tool_hooks:
        try:
            result = await hook(tool_name, tool_args, user_id)
            if result is not None:
                return result
        except Exception as e:
            logger.error(f"Pre-tool hook {hook.__name__} failed: {e}")
            # Don't block on hook errors — fail open
    return None


async def run_post_tool_hooks(
    tool_name: str,
    result: Any,
    duration_ms: float,
    user_id: str = ""
) -> None:
    """Run all registered post-tool hooks. Errors are logged, not raised."""
    for hook in _post_tool_hooks:
        try:
            await hook(tool_name, result, duration_ms, user_id)
        except Exception as e:
            logger.error(f"Post-tool hook {hook.__name__} failed: {e}")


# ─── Helper for tracking call duration ───────────────────────────────────────

class ToolTimer:
    """Context manager to measure tool execution time."""
    def __init__(self):
        self.start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.monotonic() - self.start) * 1000
