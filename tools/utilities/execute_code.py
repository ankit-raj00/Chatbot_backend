"""
execute_code tool — sandboxed Python code execution in a subprocess.

Features:
- Runs code in a subprocess with a configurable timeout (60s)
- Captures stdout + stderr
- Auto-pip-installs missing packages and retries once
- Cross-platform workspace (Windows + Linux/Mac)
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

# Cross-platform workspace: env var > home dir fallback (no /tmp on Windows)
_DEFAULT_WORKSPACE = str(Path.home() / "agentx_workspace")
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", _DEFAULT_WORKSPACE))
_TIMEOUT = 60  # seconds


def _workspace_for(user_id: str = "anonymous") -> Path:
    """Return (and create) the per-user workspace directory."""
    ws = WORKSPACE_ROOT / user_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


async def _exec_code(code: str, cwd: str, timeout: int = _TIMEOUT) -> str:
    """Write code to a temp file and run it in a subprocess."""
    # Ensure directory exists
    Path(cwd).mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=cwd, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"TIMEOUT: Code ran for more than {timeout}s and was killed."

        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")
        combined = (out + err).strip()
        return combined[:10000] if combined else "(Script ran with no output)"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _auto_install_and_retry(code: str, error: str, cwd: str) -> str | None:
    """If a ModuleNotFoundError occurred, try to pip-install the missing package and re-run."""
    if "ModuleNotFoundError: No module named" not in error:
        return None
    try:
        pkg = error.split("No module named '")[1].split("'")[0].split(".")[0]
        # Map common package names
        pkg_map = {"fpdf": "fpdf2", "docx": "python-docx", "pptx": "python-pptx"}
        pkg = pkg_map.get(pkg, pkg)
    except IndexError:
        return None

    try:
        install_proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", pkg, "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(install_proc.communicate(), timeout=120)
    except Exception:
        return None
    
    # Retry execution after install
    return await _exec_code(code, cwd)


def make_execute_code_tool(workspace_dir: str):
    """
    Factory: returns an execute_code LangChain tool bound to a specific workspace directory.
    This avoids exposing user_id as a parameter in the LLM tool schema.
    """
    @tool
    async def execute_code(code: str) -> str:
        """
        Execute Python code in a sandboxed subprocess and return the output (stdout + stderr).

        Use this to:
        - Create PDF files with reportlab or fpdf2
        - Create Word documents with python-docx
        - Create PowerPoint files with python-pptx
        - Create Excel files with openpyxl
        - Run data analysis with pandas/matplotlib
        - Do any computation or file processing

        Files created by the script are saved to the workspace directory.
        If a required library is missing, it will be auto-installed and the script retried.

        Args:
            code: A complete, runnable Python script (not a snippet — must be self-contained)
        """
        result = await _exec_code(code, workspace_dir)
        
        # Auto-install missing packages and retry once
        if "ModuleNotFoundError" in result or "No module named" in result:
            retry = await _auto_install_and_retry(code, result, workspace_dir)
            if retry is not None:
                return f"[Auto-installed missing package and retried]\n{retry}"
        
        return result

    return execute_code


# Default global instance using anonymous workspace (for registry / testing)
@tool
async def execute_code(code: str) -> str:
    """
    Execute Python code in a sandboxed subprocess and return the output (stdout + stderr).

    Use this to create PDFs, Word docs, Excel files, run analysis, or any Python task.
    Files are saved to the workspace. Missing libraries are auto-installed.

    Args:
        code: A complete, runnable Python script
    """
    cwd = str(_workspace_for("anonymous"))
    result = await _exec_code(code, cwd)
    if "ModuleNotFoundError" in result or "No module named" in result:
        retry = await _auto_install_and_retry(code, result, cwd)
        if retry is not None:
            return f"[Auto-installed missing package and retried]\n{retry}"
    return result
