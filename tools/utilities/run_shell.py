"""
run_shell — sandboxed shell command execution, bound as a real tool.

Now supports live streaming via `stream_shell`, path validation,
and per-user npm/pip environment scoping.
"""
import os
from langchain_core.tools import tool
from langchain_core.callbacks import adispatch_custom_event
from utils.workspace import conversation_workspace_for, pip_cache_dir_for, npm_prefix_for
from utils.code_executor import stream_shell, sandbox_env
from utils import sandbox_client
from utils.hooks import check_run_shell_command, BLOCKED_PATTERNS


def make_run_shell_tool(user_id: str, conversation_id: str):
    cwd = str(conversation_workspace_for(user_id, conversation_id))

    @tool
    async def run_shell(command: str) -> str:
        """
        Execute a shell command inside the user's sandboxed workspace
        directory and return combined stdout+stderr.

        All operations happen in a sandboxed per-user directory — you
        cannot access system directories. Destructive commands
        (rm -rf, sudo, fork bombs, curl|sh, etc.) are blocked.
        Path traversals (e.g. ../../otheruser) are blocked.

        Use for: listing/inspecting files, running scripts you've already
        written, checking command output, exploring the workspace.

        Args:
            command: The shell command to execute.
        """
        # Centralized in utils/hooks.py (Tier 2.3) — the SAME check also runs
        # automatically as a pre-tool hook for every dispatch through
        # agent_tool_node; calling it here too means this tool still enforces
        # itself even when invoked directly (e.g. tests), not just via the
        # agent's normal dispatch path.
        block_reason = check_run_shell_command(command, user_id, conversation_id)
        if block_reason:
            return f"BLOCKED: {block_reason}"

        if sandbox_client.is_remote():
            stream = sandbox_client.stream_shell_remote(user_id, conversation_id, command, 120)
        else:
            # Prepare isolated environment. SECURITY: build off sandbox_env()'s safe
            # allowlist, never {**os.environ} — the backend's real environment holds
            # every service credential (QDRANT_API_KEY, MONGO_URI, GOOGLE_API_KEY,
            # JWT_SECRET_KEY, ...) and sandboxed user code must never be able to read
            # them (this was a confirmed exploitable leak — see code_executor.py).
            npm_prefix = npm_prefix_for(user_id)
            env = sandbox_env({
                "PIP_CACHE_DIR": str(pip_cache_dir_for(user_id)),
                "NPM_CONFIG_PREFIX": str(npm_prefix),
            })
            env["PATH"] = f"{npm_prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}"
            stream = stream_shell(command, cwd, timeout=120,
                                  blocked_patterns=BLOCKED_PATTERNS, env=env)

        lines = []
        async for item in stream:
            if "line" in item:
                lines.append(item["line"])
                await adispatch_custom_event(
                    "exec_output",
                    {"tool": "run_shell", "line": item["line"], "stream": item["stream"]},
                )
                
        return "\n".join(lines) or "(no output)"

    return run_shell
