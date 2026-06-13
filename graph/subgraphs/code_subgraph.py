"""
Code Subgraph — code generation, debugging, review, and execution.

AFC-safe: Uses prompt-based code extraction instead of bind_tools.
The LLM writes Python code in ```python ... ``` blocks.
We extract and execute it with subprocess, then feed results back.
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

MAX_ITER = 8
from utils.workspace import workspace_for as _workspace_for
from utils.code_executor import run_python as _run_python, auto_install_and_retry as _auto_install_and_retry, extract_python_blocks as _extract_python_blocks

CODE_SYSTEM = """You are AgentX Code Agent — a specialist in software development.

You write high-quality code, debug issues, and verify your work by executing it.

## HOW TO VERIFY CODE
When you write Python code that can be run, put it in a ```python ... ``` block.
I will execute it and show you the output so you can verify it works.
If there's an error, fix it and put the corrected code in a new ```python ... ``` block.

## RULES
- Always write complete, production-ready code — not pseudocode
- Include error handling
- For frontend/HTML/CSS code, just write it (no execution needed)
- For Python code, verify it runs correctly

Follow ACTIVE SKILL instructions if present."""


async def code_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    try:
        from graph.llm_registry import get_llm

        model      = state.get("model", "gemini-2.5-flash")
        skill_body = state.get("skill_body", "")
        user_id    = state.get("user_id", "anonymous")

        workspace = _workspace_for(user_id)
        cwd = str(workspace)

        system_content = CODE_SYSTEM
        if skill_body:
            system_content += f"\n\n## ACTIVE SKILL\n{skill_body}"

        messages = list(state["messages"])
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=system_content)
        else:
            messages = [SystemMessage(content=system_content)] + messages

        llm = get_llm(model)  # NO bind_tools — AFC-safe

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

            code_blocks = _extract_python_blocks(resp_text)

            if not code_blocks:
                # Pure text response — done
                if resp_text.strip():
                    logger.info(f"code_subgraph.done iter={iteration}")
                    return {"messages": [response], "final_response": resp_text}
                continue

            # Execute code to verify
            code = code_blocks[0]
            output = await _run_python(code, cwd)
            logger.info(f"code_subgraph.exec iter={iteration} output={output[:100]!r}")

            if any(e in output for e in ["Traceback", "Error:", "TIMEOUT", "SyntaxError"]):
                messages.append(HumanMessage(
                    content=f"The code produced an error:\n```\n{output}\n```\nPlease fix it."
                ))
            else:
                # Success — feed output back for final response
                messages.append(HumanMessage(
                    content=f"Code executed successfully. Output:\n```\n{output}\n```\nPresent the solution to the user."
                ))
                final = await llm.ainvoke(messages)
                final_text = ""
                if isinstance(final.content, list):
                    final_text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in final.content)
                else:
                    final_text = str(final.content or "")
                logger.info(f"code_subgraph.done success iter={iteration}")
                return {"messages": [final], "final_response": final_text}

        # Return last AI response
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                text = str(msg.content) if not isinstance(msg.content, list) else \
                       "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in msg.content)
                if text.strip():
                    return {"messages": [msg], "final_response": text}

        fallback = "I was unable to complete the task. Please try again."
        return {"messages": [AIMessage(content=fallback)], "final_response": fallback}

    except Exception as e:
        logger.error(f"code_subgraph.crash: {e}", exc_info=True)
        error_text = f"Error in code agent: {str(e)}"
        return {"messages": [AIMessage(content=error_text)], "final_response": error_text}
