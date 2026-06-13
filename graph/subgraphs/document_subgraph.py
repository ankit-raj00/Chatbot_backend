"""
Document Subgraph for AgentX
Handles generating professional documents (.pdf, .docx, .pptx, .xlsx) via Python scripts.
"""

import asyncio
import os
import sys
import tempfile
import json
from pathlib import Path
import logging

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from graph.supervisor import SupervisorState

from utils.workspace import workspace_for as _workspace_for
from utils.code_executor import run_python as _run_python, auto_install_and_retry as _auto_install_and_retry, extract_python_blocks as _extract_python_blocks

logger = logging.getLogger(__name__)

MAX_ITER = 3


def _build_document_system(user_id: str, cwd: str, skill_body: str = "") -> str:
    base = f"""You are AgentX Document Agent — a specialist in creating professional documents.

## YOUR WORKFLOW
When asked to create a document, you must:
1. Write a complete Python script inside a ```python ... ``` code block
2. I will execute it and show you the output
3. If there's an error, write a corrected script (again in a ```python``` block)
4. Repeat until the document is created successfully
5. Then write a friendly final message confirming what was created

## KEY LIBRARIES (will be auto-installed if missing)
- PDF:          `from reportlab.pdfgen import canvas` or `from fpdf import FPDF`
- Word (.docx): `from docx import Document`
- PowerPoint:   `from pptx import Presentation`
- Excel:        `import openpyxl`

## CRITICAL RULES
- Your Python script MUST print the output filename at the end (e.g. `print("Saved: Dogs_Report.pdf")`)
- Save ALL files to the CURRENT DIRECTORY using just the filename, never an absolute path
- Bad:  `with open("/home/user/Dogs_Report.pdf", "wb") as f:`
- Good: `with open("Dogs_Report.pdf", "wb") as f:`
- When the script succeeds, write a friendly summary and include this EXACT download link format:
  [Download filename.pdf](/api/outputs/my/filename.pdf)
  Replace "filename.pdf" with the EXACT filename your script printed.
- Make documents rich and comprehensive — not placeholder content
- Never apologize — just write the code and fix errors when they occur

## WORKSPACE
Current working directory (save files HERE): {cwd}
"""
    if skill_body:
        base += f"\n\n## ACTIVE SKILL INSTRUCTIONS\n{skill_body}"
    return base


async def document_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    try:
        from graph.llm_registry import get_llm

        model      = state.get("model", "gemini-2.5-flash")
        skill_body = state.get("skill_body", "")
        user_id    = state.get("user_id", "anonymous")

        workspace = _workspace_for(user_id)
        cwd = str(workspace)

        # BUILD SYSTEM PROMPT DYNAMICALLY
        system_content = _build_document_system(user_id, cwd, skill_body)

        messages = list(state.get("messages", []))
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=system_content)
        else:
            messages = [SystemMessage(content=system_content)] + messages

        llm = get_llm(model)

        for iteration in range(MAX_ITER):
            response = await llm.ainvoke(messages)
            messages.append(response)

            if isinstance(response.content, list):
                resp_text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in response.content
                )
            else:
                resp_text = str(response.content or "")

            logger.info(f"document_subgraph.iter={iteration} response_len={len(resp_text)}")

            code_blocks = _extract_python_blocks(resp_text)

            if not code_blocks:
                if resp_text.strip():
                    logger.info(f"document_subgraph.done no_code iter={iteration}")
                    return {"messages": [response], "final_response": resp_text}
                else:
                    messages.append(HumanMessage(
                        content="Please write the Python script now in a ```python ... ``` code block."
                    ))
                    continue

            code = code_blocks[0]
            logger.info(f"document_subgraph.executing_code iter={iteration} code_len={len(code)}")
            output = await _run_python(code, cwd)

            # Auto-install missing packages and retry
            if "No module named" in output or "ModuleNotFoundError" in output:
                retry = await _auto_install_and_retry(code, output, cwd)
                if retry is not None:
                    output = f"[Package installed, retried]\n{retry}"

            logger.info(f"document_subgraph.code_output iter={iteration} output={output[:200]!r}")

            if any(err in output for err in ["Traceback", "Error:", "TIMEOUT", "SyntaxError"]):
                messages.append(HumanMessage(
                    content=f"The script produced an error:\n```\n{output}\n```\nPlease fix the script and try again."
                ))
                continue

            # Script succeeded
            messages.append(HumanMessage(
                content=f"The script ran successfully. Output:\n```\n{output}\n```\nNow write a friendly message to the user confirming what was created, and include the download link in this format: [Download filename](/api/outputs/my/filename)"
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

        fallback = "I was unable to create the document after multiple attempts. Please try rephrasing your request."
        logger.warning(f"document_subgraph.max_iter_reached iterations={MAX_ITER}")
        return {"messages": [AIMessage(content=fallback)], "final_response": fallback}

    except Exception as e:
        logger.error(f"document_subgraph.crash: {e}", exc_info=True)
        error_text = f"An error occurred while creating the document: {str(e)}"
        return {"messages": [AIMessage(content=error_text)], "final_response": error_text}
