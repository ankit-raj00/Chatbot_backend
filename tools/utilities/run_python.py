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
from utils.workspace import conversation_workspace_for, venv_python_for, pip_cache_dir_for, is_path_within_conversation_sandbox
from utils.code_executor import stream_python, sandbox_env, _sandbox_user_kwargs

_TIMEOUT = 300  # seconds


def _find_local_module(cwd: str, module_name: str) -> str | None:
    """Best-effort: does <module_name>.py or <module_name>/__init__.py exist
    anywhere under this conversation's workspace? If so, the real problem is
    a missing sys.path entry, not a missing PyPI package — auto-installing
    would otherwise pull an arbitrary PyPI package (whatever happens to be
    published under a name that collides with the agent's own filename)
    into the user's persistent venv unprompted. Confirmed misfire: a script
    imported a sibling file it had just saved as `file_c.py`, and this
    function tried `pip install file_c` before falling back to fixing
    sys.path."""
    root = Path(cwd)
    if not root.exists():
        return None
    for candidate in root.rglob(f"{module_name}.py"):
        return str(candidate.relative_to(root))
    for candidate in root.rglob(f"{module_name}/__init__.py"):
        return str(candidate.parent.relative_to(root))
    return None


async def _auto_install_and_retry_streaming(code: str, error: str, cwd: str, user_id: str,
                                             script_path: str | None) -> str | None:
    """Returns a fully-formed message to show the agent (already includes
    whatever prefix/explanation is appropriate), or None if `error` wasn't a
    'No module named' error at all — callers should leave the original
    output untouched in that case."""
    if "No module named" not in error:
        return None
    try:
        pkg_raw = error.split("No module named '")[1].split("'")[0].split(".")[0]
        pkg_map = {"fpdf": "fpdf2", "docx": "python-docx", "pptx": "python-pptx", "bs4": "beautifulsoup4"}
        pkg = pkg_map.get(pkg_raw, pkg_raw)
    except IndexError:
        return None

    local_match = await asyncio.to_thread(_find_local_module, cwd, pkg_raw)
    if local_match:
        return (f"'{pkg_raw}' looks like a local file ({local_match}) in this workspace, not a "
                f"PyPI package — not auto-installing it. Add its directory to sys.path, or "
                f"import/run it as a relative module instead.")

    # venv already exists by now (run_python built it) — resolve off the loop anyway.
    venv_python = await asyncio.to_thread(venv_python_for, user_id)
    pip_path = str(venv_python.parent / ("pip.exe" if os.name == "nt" else "pip"))
    cache_dir = str(pip_cache_dir_for(user_id))

    install_proc = await asyncio.create_subprocess_exec(
        pip_path, "install", pkg, "--quiet", "--cache-dir", cache_dir,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=sandbox_env({"PIP_CACHE_DIR": cache_dir}),
        **_sandbox_user_kwargs(),
    )
    _, install_err = await install_proc.communicate()
    if install_proc.returncode != 0:
        return (f"Failed to auto-install '{pkg}' (pip exit {install_proc.returncode}): "
                f"{install_err.decode('utf-8', 'replace')[:500]}")

    lines = []
    python_exec = str(venv_python)
    async for item in stream_python(code, cwd, timeout=_TIMEOUT, python_executable=python_exec, script_path=script_path):
        if "line" in item:
            lines.append(item["line"])
            await adispatch_custom_event(
                "exec_output",
                {"tool": "run_python", "line": item["line"], "stream": item["stream"]},
            )

    joined = "\n".join(lines) or "(no output)"
    return f"[Auto-installed missing package and retried]\n{joined}"


def make_run_python_tool(user_id: str, conversation_id: str):
    """
    Factory: returns a run_python tool bound to a specific user's workspace.

    The (possibly slow, first-time) venv creation is deferred to the first actual
    tool call and run in a thread, so building the graph / tool list never blocks
    the event loop.
    """
    cwd = str(conversation_workspace_for(user_id, conversation_id))

    @tool
    async def run_python(code: str, filename: str = "") -> str:
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
        - PREFER passing `filename` (e.g. "work/build_report.py") for anything
          beyond a one-line check. This saves the script as a real file instead
          of a throwaway one, so if it errors or needs a tweak you fix it with
          edit_file and re-run it via run_shell — instead of resending the
          entire script here again. Without `filename`, the script is discarded
          after running and can't be edited afterward.

        Args:
            code: A complete, runnable Python script (not a snippet).
            filename: Optional sandbox-relative path (e.g. "work/script.py") to
                save the script to permanently instead of discarding it after
                the run. Strongly recommended for anything you might need to
                iterate on.
        """
        # Resolve (and lazily build) the per-user venv off the event loop.
        python_executable = str(await asyncio.to_thread(venv_python_for, user_id))

        script_path = None
        if filename:
            if not is_path_within_conversation_sandbox(user_id, conversation_id, filename):
                return "BLOCKED: filename path outside sandbox"
            script_path = str(Path(cwd) / filename)

        lines = []
        async for item in stream_python(code, cwd, timeout=_TIMEOUT, python_executable=python_executable,
                                         script_path=script_path):
            if "line" in item:
                lines.append(item["line"])
                await adispatch_custom_event(
                    "exec_output",
                    {"tool": "run_python", "line": item["line"], "stream": item["stream"]},
                )

        output = "\n".join(lines) or "(no output)"
        if "ModuleNotFoundError" in output or "No module named" in output:
            retry = await _auto_install_and_retry_streaming(code, output, cwd, user_id, script_path)
            if retry is not None:
                return retry
        if filename:
            output = f"[Saved to {filename}]\n{output}"
        return output[:10000]

    return run_python
