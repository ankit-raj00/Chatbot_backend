"""
run_python — the agent's general-purpose Python sandbox.

Now supports live streaming via `stream_python` and `adispatch_custom_event`,
and uses per-user isolated virtual environments.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.callbacks import adispatch_custom_event
from utils.workspace import workspace_for, venv_python_for, pip_cache_dir_for
from utils.code_executor import stream_python

_TIMEOUT = 300  # seconds


async def _auto_install_and_retry_streaming(code: str, error: str, cwd: str, user_id: str) -> str | None:
    if "No module named" not in error:
        return None
    try:
        pkg_raw = error.split("No module named '")[1].split("'")[0].split(".")[0]
        pkg_map = {"fpdf": "fpdf2", "docx": "python-docx", "pptx": "python-pptx", "bs4": "beautifulsoup4"}
        pkg = pkg_map.get(pkg_raw, pkg_raw)
    except IndexError:
        return None
        
    pip_path = str(venv_python_for(user_id).parent / ("pip.exe" if os.name == "nt" else "pip"))
    cache_dir = str(pip_cache_dir_for(user_id))
    
    install_proc = await asyncio.create_subprocess_exec(
        pip_path, "install", pkg, "--quiet", "--cache-dir", cache_dir,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await install_proc.wait()
    
    lines = []
    python_exec = str(venv_python_for(user_id))
    async for item in stream_python(code, cwd, timeout=_TIMEOUT, python_executable=python_exec):
        if "line" in item:
            lines.append(item["line"])
            await adispatch_custom_event(
                "exec_output",
                {"tool": "run_python", "line": item["line"], "stream": item["stream"]},
            )
            
    return "\n".join(lines) or "(no output)"


def make_run_python_tool(user_id: str):
    """
    Factory: returns a run_python tool bound to a specific user's workspace.
    """
    cwd = str(workspace_for(user_id))
    python_executable = str(venv_python_for(user_id))

    @tool
    async def run_python(code: str) -> str:
        """
        Execute a complete, self-contained Python script in a sandboxed
        subprocess and return combined stdout+stderr.

        Use this for:
        - Data analysis (pandas, numpy)
        - Generating PDF (reportlab, fpdf2), DOCX (python-docx),
          PPTX (python-pptx), XLSX (openpyxl)
        - Any computation, file read/write, or transformation

        IMPORTANT:
        - Save all output files to outputs/ using relative paths, e.g. open("outputs/report.pdf", "wb").
        - The script MUST print the output filename at the end if it creates a file.
        - Missing packages are auto-installed and the script is retried once.

        Args:
            code: A complete, runnable Python script (not a snippet).
        """
        lines = []
        async for item in stream_python(code, cwd, timeout=_TIMEOUT, python_executable=python_executable):
            if "line" in item:
                lines.append(item["line"])
                await adispatch_custom_event(
                    "exec_output",
                    {"tool": "run_python", "line": item["line"], "stream": item["stream"]},
                )

        output = "\n".join(lines) or "(no output)"
        if "ModuleNotFoundError" in output or "No module named" in output:
            retry = await _auto_install_and_retry_streaming(code, output, cwd, user_id)
            if retry is not None:
                return f"[Auto-installed missing package and retried]\n{retry}"
        return output[:10000]

    return run_python
