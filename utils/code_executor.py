"""
Shared Python code execution utility for all agent subgraphs.
Provides sandboxed subprocess execution with auto-install support.
"""
import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path


async def run_python(code: str, cwd: str, timeout: int = 300) -> str:
    """
    Execute a Python script string in a subprocess.
    Returns combined stdout + stderr, truncated to 8000 chars.
    """
    Path(cwd).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=cwd, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"TIMEOUT: Script exceeded {timeout}s."
        return ((out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip() or "(no output)")[:8000]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


async def auto_install_and_retry(code: str, error: str, cwd: str) -> str | None:
    """
    If error is a ModuleNotFoundError, install the missing package and retry.
    Returns new output if install+retry succeeded, None if not a package error.
    """
    if "No module named" not in error:
        return None
    try:
        pkg_raw = error.split("No module named '")[1].split("'")[0].split(".")[0]
        pkg_map = {"fpdf": "fpdf2", "docx": "python-docx", "pptx": "python-pptx"}
        pkg = pkg_map.get(pkg_raw, pkg_raw)
    except IndexError:
        return None
    
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", pkg, "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
        return await run_python(code, cwd)
    except Exception:
        return None


def extract_python_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` code blocks from text."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


async def run_shell(cmd: str, cwd: str, blocked_patterns: list[str] | None = None) -> str:
    """
    Execute a shell command in a subprocess.
    blocked_patterns: list of substrings that block the command entirely.
    """
    if blocked_patterns:
        cl = cmd.lower()
        if any(b in cl for b in blocked_patterns):
            return "BLOCKED: Command contains a forbidden pattern."
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        return ((out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip() or "(no output)")[:8000]
    except asyncio.TimeoutError:
        return "TIMEOUT: Command exceeded 30s."
    except Exception as e:
        return f"ERROR: {e}"
