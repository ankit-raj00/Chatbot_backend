"""
Compat shim for shell_subgraph.py to preserve test imports.
"""
from tools.utilities.run_shell import BLOCKED_PATTERNS
from utils.code_executor import run_shell as _run_cmd

def _is_blocked(cmd: str) -> bool:
    cmd_lower = cmd.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return True
    return False
