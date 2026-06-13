"""
Shell Subgraph — executes shell commands inside a sandboxed workspace.

AFC-safe: No bind_tools. Instead uses prompt-based command extraction.
LLM writes shell commands in ```bash ... ``` or ```shell ... ``` blocks.
We extract and execute them with asyncio subprocess, feed results back.

Safety:
- All commands run in WORKSPACE_ROOT/{user_id}/
- Blocked patterns: rm -rf /, sudo rm, etc.
"""

import os
import re
import asyncio
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.supervisor import SupervisorState
import structlog

logger = structlog.get_logger(__name__)

MAX_ITER = 8
from utils.workspace import workspace_for as _workspace_for
from utils.code_executor import run_shell as _run_shell

# Backward compatibility alias for tests
_run_cmd = _run_shell


def _extract_shell_blocks(text: str) -> list[str]:
    """Extract ```bash or ```shell or ```sh code blocks."""
    pattern = r"```(?:bash|shell|sh|cmd|powershell)\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        # Fallback: plain ``` blocks
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return blocks


SHELL_SYSTEM = """You are AgentX Shell Agent — a specialist in file system and command-line operations.

## HOW TO EXECUTE COMMANDS
Write shell commands in ```bash ... ``` code blocks. I will execute them and show you the output.
If a command fails, read the error and write a corrected command in a new code block.

## WORKSPACE
All operations happen in a sandboxed workspace directory. You cannot access system directories.

## RULES
- Write one set of commands at a time in a code block
- Chain related commands with && or ;
- After commands succeed, write a summary for the user in plain text
- Never use destructive commands (rm -rf, sudo, etc.)

Follow ACTIVE SKILL instructions if present."""


async def shell_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    try:
        from graph.llm_registry import get_llm

        model      = state.get("model", "gemini-2.5-flash")
        skill_body = state.get("skill_body", "")
        user_id    = state.get("user_id", "anonymous")

        workspace = _workspace_for(user_id)
        cwd = str(workspace)

        system_content = SHELL_SYSTEM
        if skill_body:
            system_content += f"\n\n## SKILL INSTRUCTIONS\n{skill_body}"
        system_content += f"\n\n## WORKSPACE PATH: {cwd}"

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

            shell_blocks = _extract_shell_blocks(resp_text)

            if not shell_blocks:
                if resp_text.strip():
                    logger.info(f"shell_subgraph.done iter={iteration}")
                    return {"messages": [response], "final_response": resp_text}
                continue

            for cmd_block in shell_blocks:
                # Run each line as separate command or whole block
                cmd = cmd_block.strip()
                output = await _run_shell(cmd, cwd, blocked_patterns=[
    "rm -rf /", "rm -rf ~", "sudo rm", ":(){:|:&};:", "mkfs",
    "dd if=/dev/zero", "chmod -R 777 /", "> /dev/sda",
    "curl | sh", "wget | sh", "curl | bash", "wget | bash",
])
                logger.info(f"shell_subgraph.exec iter={iteration} cmd={cmd[:60]!r} out={output[:80]!r}")

                if any(e in output for e in ["ERROR:", "BLOCKED:", "TIMEOUT:", "command not found", "is not recognized"]):
                    messages.append(HumanMessage(
                        content=f"Command failed:\n```\n{output}\n```\nPlease try a different approach."
                    ))
                else:
                    messages.append(HumanMessage(
                        content=f"Command output:\n```\n{output}\n```\nContinue or write a summary for the user."
                    ))

            continue  # Let LLM respond to the output

        # Return last AI text response
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                text = str(msg.content) if not isinstance(msg.content, list) else \
                       "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in msg.content)
                if text.strip():
                    return {"messages": [msg], "final_response": text}

        fallback = "I was unable to complete the shell task. Please try again."
        return {"messages": [AIMessage(content=fallback)], "final_response": fallback}

    except Exception as e:
        logger.error(f"shell_subgraph.crash: {e}", exc_info=True)
        error_text = f"Error in shell agent: {str(e)}"
        return {"messages": [AIMessage(content=error_text)], "final_response": error_text}
