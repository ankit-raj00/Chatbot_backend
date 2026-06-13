"""
Document Subgraph — generates PDF, DOCX, PPTX, Excel, and diagram files.

Strategy (AFC-safe):
  Instead of bind_tools (which triggers Gemini AFC and causes empty responses),
  we use a prompt-based code extraction loop:

    1. Ask the LLM to write a Python script in a ```python ... ``` block
    2. Extract the code with regex
    3. Execute it with subprocess (no tool calling needed)
    4. Feed the output back → LLM writes the final user-facing response

  This completely bypasses Gemini's Automatic Function Calling (AFC) issue.
"""

import os
import re
import sys
import asyncio
import tempfile
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.supervisor import SupervisorState
import structlog

logger = structlog.get_logger(__name__)

MAX_ITER = 5          # Max code-write-execute cycles
_DEFAULT_WS = str(Path.home() / "agentx_workspace")


def _workspace_for(user_id: str = "anonymous") -> Path:
    ws = Path(os.getenv("WORKSPACE_ROOT", _DEFAULT_WS)) / user_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


async def _run_python(code: str, cwd: str, timeout: int = 300) -> str:
    """Execute a Python script string in a subprocess. Returns stdout+stderr."""
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
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"TIMEOUT: Script exceeded {timeout}s."

        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")
        return ((out + err).strip() or "(no output)")[:8000]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


async def _auto_install_and_retry(code: str, error: str, cwd: str) -> str:
    """Auto-install missing package from ModuleNotFoundError and re-run."""
    if "No module named" not in error:
        return error

    try:
        pkg_raw = error.split("No module named '")[1].split("'")[0].split(".")[0]
        pkg_map = {"fpdf": "fpdf2", "docx": "python-docx", "pptx": "python-pptx"}
        pkg = pkg_map.get(pkg_raw, pkg_raw)
        logger.info(f"document_subgraph.auto_install pkg={pkg}")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", pkg, "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
        return await _run_python(code, cwd)
    except Exception as e:
        return f"{error}\n[Auto-install failed: {e}]"


def _extract_python_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` code blocks from text."""
    pattern = r"```python\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


DOCUMENT_SYSTEM = """You are AgentX Document Agent — a specialist in creating professional documents.

## YOUR WORKFLOW
When asked to create a document, you must:
1. Write a complete Python script inside a ```python ... ``` code block
2. I will execute it and show you the output
3. If there's an error, write a corrected script (again in a ```python``` block)
4. Repeat until the document is created successfully
5. Then write a friendly final message confirming what was created

## KEY LIBRARIES (will be auto-installed if missing)
- PDF:          `from fpdf import FPDF`  OR  `from reportlab.pdfgen import canvas`
- Word (.docx): `from docx import Document`
- PowerPoint:   `from pptx import Presentation`
- Excel:        `import openpyxl`

## IMPORTANT
- Your Python script MUST print the output filename (e.g. `print("Saved: Dogs_Report.pdf")`)
- When the script succeeds, write a friendly summary and INCLUDE A DOWNLOAD LINK like this:
  `[Download {filename}](/api/outputs/download/{user_id}/{filename})`
- Make documents rich and comprehensive — not placeholder content
- Never apologize — just write the code and fix errors when they occur
- Follow ACTIVE SKILL instructions below if present"""


async def document_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    try:
        from graph.llm_registry import get_llm

        model      = state.get("model", "gemini-2.5-flash")
        skill_body = state.get("skill_body", "")
        user_id    = state.get("user_id", "anonymous")

        workspace = _workspace_for(user_id)
        cwd = str(workspace)

        # ── System prompt ────────────────────────────────────────────────────
        system_content = DOCUMENT_SYSTEM
        if skill_body:
            system_content += f"\n\n## ACTIVE SKILL INSTRUCTIONS\n{skill_body}"
        system_content += f"\n\n## WORKSPACE\nSave all files to the current directory. Workspace path: {cwd}"

        messages = list(state["messages"])
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=system_content)
        else:
            messages = [SystemMessage(content=system_content)] + messages

        llm = get_llm(model)   # NO bind_tools — bypasses AFC completely

        # ── Prompt-based code extraction loop ────────────────────────────────
        for iteration in range(MAX_ITER):
            response = await llm.ainvoke(messages)
            messages.append(response)

            # Extract text from response
            if isinstance(response.content, list):
                resp_text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in response.content
                )
            else:
                resp_text = str(response.content or "")

            logger.info(f"document_subgraph.iter={iteration} response_len={len(resp_text)}")

            # Find python code blocks
            code_blocks = _extract_python_blocks(resp_text)

            if not code_blocks:
                # No code block — LLM gave a text response (done or needs more info)
                if resp_text.strip():
                    logger.info(f"document_subgraph.done no_code iter={iteration}")
                    return {"messages": [response], "final_response": resp_text}
                else:
                    # Empty response — prompt it again
                    messages.append(HumanMessage(
                        content="Please write the Python script now in a ```python ... ``` code block."
                    ))
                    continue

            # Execute the first code block (usually there's only one)
            code = code_blocks[0]
            logger.info(f"document_subgraph.executing_code iter={iteration} code_len={len(code)}")
            output = await _run_python(code, cwd)

            # Auto-install missing packages
            if "No module named" in output:
                output = await _auto_install_and_retry(code, output, cwd)

            logger.info(f"document_subgraph.code_output iter={iteration} output={output[:200]!r}")

            # If script failed, feed the error back so LLM can fix it
            if any(err in output for err in ["Traceback", "Error:", "TIMEOUT", "SyntaxError"]):
                messages.append(HumanMessage(
                    content=f"The script produced an error:\n```\n{output}\n```\nPlease fix the script and try again."
                ))
                continue

            # Script succeeded! Ask LLM to write the user-facing summary
            messages.append(HumanMessage(
                content=f"The script ran successfully. Output:\n```\n{output}\n```\nNow write a friendly message to the user confirming what was created and where they can find the file."
            ))
            final = await llm.ainvoke(messages)
            if isinstance(final.content, list):
                final_text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in final.content
                )
            else:
                final_text = str(final.content or "")

            logger.info(f"document_subgraph.done success iter={iteration}")
            return {"messages": [final], "final_response": final_text}

        # Exhausted all iterations
        fallback = "I was unable to create the document after multiple attempts. Please try rephrasing your request."
        logger.warning(f"document_subgraph.max_iter_reached iterations={MAX_ITER}")
        return {"messages": [AIMessage(content=fallback)], "final_response": fallback}

    except Exception as e:
        logger.error(f"document_subgraph.crash: {e}", exc_info=True)
        error_text = f"An error occurred while creating the document: {str(e)}"
        return {"messages": [AIMessage(content=error_text)], "final_response": error_text}
