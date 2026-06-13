"""
run_python — the agent's general-purpose Python sandbox.

Replaces the prompt-based ```python ... ``` extraction used by
code_subgraph / document_subgraph / data_subgraph / shell_subgraph.
Now a first-class bind_tools-compatible tool.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from utils.workspace import workspace_for
from utils.code_executor import run_python as _run_python, auto_install_and_retry as _auto_install_and_retry

_TIMEOUT = 300  # seconds — matches utils/code_executor.run_python default


def make_run_python_tool(user_id: str):
    """
    Factory: returns a run_python tool bound to a specific user's workspace.
    Called per-request in agent_node so user_id isn't exposed in the LLM schema.
    """
    cwd = str(workspace_for(user_id))

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
        - Save all output files to the CURRENT DIRECTORY using just the
          filename (e.g. open("report.pdf", "wb")) — never absolute paths.
        - The script MUST print the output filename at the end if it
          creates a file, e.g. print("Saved: report.pdf")
        - Missing packages are auto-installed and the script is retried once.

        Args:
            code: A complete, runnable Python script (not a snippet).
        """
        output = await _run_python(code, cwd, timeout=_TIMEOUT)
        if "ModuleNotFoundError" in output or "No module named" in output:
            retry = await _auto_install_and_retry(code, output, cwd)
            if retry is not None:
                return f"[Auto-installed missing package and retried]\n{retry}"
        return output

    return run_python
