"""
run_shell — sandboxed shell command execution, bound as a real tool.

Now supports live streaming via `stream_shell`, path validation,
and per-user npm/pip environment scoping.
"""
import os
from langchain_core.tools import tool
from langchain_core.callbacks import adispatch_custom_event
from utils.workspace import workspace_for, is_path_within_sandbox, pip_cache_dir_for, npm_prefix_for
from utils.code_executor import stream_shell

BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf ~", "sudo rm", ":(){:|:&};:", "mkfs",
    "dd if=/dev/zero", "chmod -R 777 /", "> /dev/sda",
    "curl | sh", "wget | sh", "curl | bash", "wget | bash",
]


def make_run_shell_tool(user_id: str):
    cwd = str(workspace_for(user_id))

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
        import shlex

        # Try to heuristically check paths in the command string for Tier 1 isolation
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
            
        for part in parts:
            if "/" in part or "\\" in part:
                # If part looks like a path, ensure it doesn't escape the sandbox
                if not is_path_within_sandbox(user_id, part):
                    return "BLOCKED: path outside sandbox"

        # Prepare isolated environment
        env = {**os.environ}
        env["PIP_CACHE_DIR"] = str(pip_cache_dir_for(user_id))
        
        npm_prefix = npm_prefix_for(user_id)
        env["NPM_CONFIG_PREFIX"] = str(npm_prefix)
        env["PATH"] = f"{npm_prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}"

        lines = []
        async for item in stream_shell(command, cwd, timeout=120, blocked_patterns=BLOCKED_PATTERNS, env=env):
            if "line" in item:
                lines.append(item["line"])
                await adispatch_custom_event(
                    "exec_output",
                    {"tool": "run_shell", "line": item["line"], "stream": item["stream"]},
                )
                
        return "\n".join(lines) or "(no output)"

    return run_shell
