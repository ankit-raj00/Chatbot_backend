"""
Hook system for agent tool lifecycle events.
Inspired by Claude Agent SDK's lifecycle hooks.

Usage — register a hook:
    from utils.hooks import register_pre_tool_hook

    @register_pre_tool_hook
    async def my_guard(tool_name: str, tool_args: dict, user_id: str,
                        conversation_id: str = "") -> dict | None:
        if tool_name == "bash" and "rm -rf" in tool_args.get("command", ""):
            return {"deny": True, "reason": "Destructive bash command blocked by security hook"}
        return None   # None means allow

Usage — in tool node:
    from utils.hooks import run_pre_tool_hooks, run_post_tool_hooks
    result = await run_pre_tool_hooks(tool_name, tool_args, user_id, conversation_id)
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

import structlog
logger = structlog.get_logger(__name__)

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
async def _log_tool_call(tool_name: str, tool_args: dict, user_id: str,
                          conversation_id: str = "") -> None:
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


# ─── Centralized sandbox guardrails (Tier 2.3, HARDENING_PLAN.md) ────────────
# Previously copy-pasted independently across run_python.py/run_shell.py/
# edit_file.py/analyze_image.py — a newly added tool got zero protection
# unless its author remembered to paste the same checks in again. Now: ONE
# implementation, used two ways —
#   1. Registered as pre-tool hooks below, so EVERY tool call dispatched
#      through agent_tool_node is checked automatically, new tools included,
#      whether or not their author remembers to call anything.
#   2. Called directly by the tool functions themselves too (see
#      tools/utilities/run_python.py etc.), so a tool invoked outside the
#      normal agent dispatch path (a direct .ainvoke() — this is exactly how
#      tests/test_agent_v3.py and tests/test_e2e.py exercise these tools)
#      still enforces it. Belt and suspenders on the SAME source of truth,
#      not two independently-maintained copies that could drift apart.

BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf ~", "sudo rm", ":(){:|:&};:", "mkfs",
    "dd if=/dev/zero", "chmod -R 777 /", "> /dev/sda",
    "curl | sh", "wget | sh", "curl | bash", "wget | bash",
    # Confirmed live: with NO restriction at all, a benign prompt led the
    # agent to reach loopback (enumerating this app's own /admin/* routes)
    # and the AWS metadata endpoint from inside run_python — this is the
    # same protection for raw shell commands (curl/wget/nc/etc.), which the
    # Python-level socket guard in code_executor.py doesn't cover since
    # they're not Python code. Simple substring match (same mechanism as
    # every other entry above), so this can over-match on an unrelated
    # command that happens to mention one of these strings (e.g. grepping a
    # log file containing "127.0.0.1") — an accepted tradeoff for a coding
    # sandbox where the agent can always take a different approach if
    # legitimately blocked. Broader ranges (RFC1918 generally) are enforced
    # instead at the host firewall and the Python guard, where over-matching
    # on short numeric substrings isn't a risk.
    "169.254.169.254", "169.254.", "127.0.0.1", "localhost", "172.17.",
]

# Which single sandbox-relative-path argument each tool takes, if any.
_SANDBOX_PATH_ARG_BY_TOOL = {
    "sandbox_run_python": "filename",   # optional — only present when persisting a script
    "sandbox_edit_file": "path",
    "sandbox_analyze_image": "sandbox_path",
}


def check_sandbox_path(tool_name: str, tool_args: dict, user_id: str,
                        conversation_id: str) -> str | None:
    """Returns a block REASON string if tool_args' path argument escapes the
    conversation sandbox, else None (allowed). Tools whose path arg is
    absent/optional (e.g. run_python without filename) are not checked here —
    there's nothing to validate."""
    arg_name = _SANDBOX_PATH_ARG_BY_TOOL.get(tool_name)
    if not arg_name:
        return None
    path = tool_args.get(arg_name)
    if not path:
        return None
    from utils.workspace import is_path_within_conversation_sandbox
    if not is_path_within_conversation_sandbox(user_id, conversation_id, path):
        return f"{arg_name} path outside sandbox: {path!r}"
    return None


def check_run_shell_command(command: str, user_id: str, conversation_id: str) -> str | None:
    """Returns a block REASON string if `command` attempts a sandbox escape
    or matches a destructive-command pattern, else None (allowed). Moved
    verbatim from run_shell.py's tool function."""
    if not command:
        return None
    if any(b in command.lower() for b in BLOCKED_PATTERNS):
        return "command contains a forbidden pattern"

    import shlex
    from utils.workspace import is_path_within_conversation_sandbox

    # Tokenize with shell OPERATORS (; & | < >) split into their own tokens,
    # so `cd ..; ls` yields ['cd', '..', ';', 'ls'] and the '..' is caught,
    # rather than a glued '..;' token that would slip past the checks below.
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        parts = list(lex)
    except ValueError:
        parts = command.split()

    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    for i, part in enumerate(parts):
        p = _norm(part)
        # Block parent-directory traversal even when the token has no other
        # slash (e.g. bare `cd ..`), which a slash-only check would miss.
        if p == ".." or p.startswith("../") or "/../" in p or p.endswith("/.."):
            return "parent-directory ('..') access is not allowed"
        if p.startswith("~"):
            return "home-directory ('~') access is not allowed"
        if ("/" in p) and not is_path_within_conversation_sandbox(user_id, conversation_id, part):
            return "path outside sandbox"
        # Guard the `cd`/`pushd`/`chdir` target explicitly (it mutates cwd for
        # every following command in the same shell invocation).
        if p.lower() in ("cd", "pushd", "chdir") and i + 1 < len(parts):
            target = parts[i + 1]
            tnorm = _norm(target)
            if (tnorm == ".." or tnorm.startswith("../") or "/../" in tnorm
                    or tnorm.startswith("~")
                    or not is_path_within_conversation_sandbox(user_id, conversation_id, target)):
                return "cannot change directory outside the sandbox"
    return None


@register_pre_tool_hook
async def _sandbox_guard_hook(tool_name: str, tool_args: dict, user_id: str,
                               conversation_id: str = "") -> dict | None:
    """The automatic safety net: fires for every tool call dispatched through
    agent_tool_node, so a newly added tool inherits this protection even if
    its author never calls check_sandbox_path/check_run_shell_command
    directly."""
    reason = check_sandbox_path(tool_name, tool_args, user_id, conversation_id)
    if reason:
        return {"deny": True, "reason": reason}
    if tool_name == "sandbox_run_shell":
        reason = check_run_shell_command(tool_args.get("command", ""), user_id, conversation_id)
        if reason:
            return {"deny": True, "reason": reason}
    return None


# ─── Runner functions ─────────────────────────────────────────────────────────

async def run_pre_tool_hooks(
    tool_name: str,
    tool_args: dict,
    user_id: str = "",
    conversation_id: str = "",
) -> dict | None:
    """
    Run all registered pre-tool hooks in order.
    Returns the first non-None result (deny or modify).
    Returns None if all hooks pass.
    """
    for hook in _pre_tool_hooks:
        try:
            result = await hook(tool_name, tool_args, user_id, conversation_id)
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
