"""
Data Subgraph — CSV/Excel/JSON analysis, statistics, and insights.

AFC-safe: No bind_tools. Data is injected directly into the context.
Python code blocks are extracted and executed via subprocess.
"""

import os
import re
import sys
import json
import asyncio
import tempfile
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.supervisor import SupervisorState
import structlog

logger = structlog.get_logger(__name__)

MAX_ITER = 6
from utils.workspace import workspace_for as _workspace_for
from utils.code_executor import run_python as _run_python, extract_python_blocks as _extract_python_blocks

async def _load_file_context(file_path: str) -> str:
    """Load a data file and return its content as a string for LLM context."""
    try:
        from services.universal_file_reader import extract_any_file
        p = Path(file_path)
        if not p.exists():
            return f"File not found: {file_path}"
        content = await extract_any_file(p)
        if isinstance(content, dict) and "error" in content:
            return f"Error reading file: {content['error']}"
        
        import pandas as pd
        if isinstance(content, dict) and content.get("type") == "csv" and "rows" in content:
            df = pd.DataFrame(content["rows"])
        elif isinstance(content, dict) and content.get("type") == "excel" and "sheets" in content:
            first_sheet = next(iter(content["sheets"].values()))
            df = pd.DataFrame(first_sheet[1:], columns=first_sheet[0]) if len(first_sheet) > 1 else pd.DataFrame()
        else:
            return json.dumps(content, ensure_ascii=False)[:5000]

        import io
        buf = io.StringIO()
        buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        buf.write(f"Columns: {list(df.columns)}\n\n")
        buf.write("Statistics:\n")
        buf.write(df.describe(include="all").to_string())
        buf.write(f"\n\nMissing values:\n{df.isnull().sum().to_string()}")
        buf.write(f"\n\nFirst 10 rows:\n{df.head(10).to_string()}")
        return buf.getvalue()[:8000]
    except Exception as e:
        return f"Error loading file: {e}"


DATA_SYSTEM = """You are AgentX Data Agent — a specialist in data analysis.

## HOW TO ANALYZE DATA
If data is provided in context, analyze it directly.
For complex analysis, write Python pandas code in a ```python ... ``` block — I will execute it and show you the output.

## RULES
1. Show exact numbers — never vague descriptions
2. When the data is provided, answer directly without code if possible
3. Use Python code blocks for complex aggregations, charts, or transformations
4. Apply ACTIVE SKILL instructions if present"""


async def data_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    try:
        from graph.llm_registry import get_llm

        model      = state.get("model", "gemini-2.5-flash")
        skill_body = state.get("skill_body", "")
        user_id    = state.get("user_id", "anonymous")
        context_files = state.get("context_files", [])

        workspace = _workspace_for(user_id)
        cwd = str(workspace)

        system_content = DATA_SYSTEM
        if skill_body:
            system_content += f"\n\n## ACTIVE SKILL INSTRUCTIONS\n{skill_body}"

        # Pre-load any attached data files into context
        file_contexts = []
        for f in (context_files or []):
            ctx = await _load_file_context(f)
            file_contexts.append(f"## File: {f}\n{ctx}")

        if file_contexts:
            system_content += "\n\n## DATA FILES\n" + "\n\n".join(file_contexts)

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
                if resp_text.strip():
                    logger.info(f"data_subgraph.done iter={iteration}")
                    return {"messages": [response], "final_response": resp_text}
                continue

            code = code_blocks[0]
            output = await _run_python(code, cwd)
            logger.info(f"data_subgraph.exec iter={iteration} output={output[:100]!r}")

            if any(e in output for e in ["Traceback", "Error:", "TIMEOUT", "SyntaxError"]):
                messages.append(HumanMessage(
                    content=f"Error running analysis:\n```\n{output}\n```\nPlease fix the code."
                ))
            else:
                messages.append(HumanMessage(
                    content=f"Analysis output:\n```\n{output}\n```\nPresent findings to the user."
                ))
                final = await llm.ainvoke(messages)
                final_text = ""
                if isinstance(final.content, list):
                    final_text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in final.content)
                else:
                    final_text = str(final.content or "")
                logger.info(f"data_subgraph.done success iter={iteration}")
                return {"messages": [final], "final_response": final_text}

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                text = str(msg.content) if not isinstance(msg.content, list) else \
                       "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in msg.content)
                if text.strip():
                    return {"messages": [msg], "final_response": text}

        fallback = "I was unable to complete the data analysis. Please try again."
        return {"messages": [AIMessage(content=fallback)], "final_response": fallback}

    except Exception as e:
        logger.error(f"data_subgraph.crash: {e}", exc_info=True)
        error_text = f"Error in data agent: {str(e)}"
        return {"messages": [AIMessage(content=error_text)], "final_response": error_text}
