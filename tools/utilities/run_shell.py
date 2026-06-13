"""
run_shell — sandboxed shell command execution, bound as a real tool.

Replaces the prompt-based ```bash ... ``` extraction in shell_subgraph.
Blocked-pattern list moved here from shell_subgraph.py (single source of truth).
"""
from langchain_core.tools import tool
from utils.workspace import workspace_for
from utils.code_executor import run_shell as _run_shell

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

        Use for: listing/inspecting files, running scripts you've already
        written, checking command output, exploring the workspace.

        Args:
            command: The shell command to execute.
        """
        return await _run_shell(command, cwd, blocked_patterns=BLOCKED_PATTERNS)

    return run_shell
