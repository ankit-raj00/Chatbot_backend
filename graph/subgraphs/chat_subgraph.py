"""
Chat Subgraph — wraps the existing chat_model_node for general conversation.
This keeps 100% backward compatibility with the old single-graph architecture.

Now includes a proper tool execution loop so user-enabled tools (roll_dice,
get_weather, execute_code, etc.) are actually invoked when the LLM calls them.
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.supervisor import SupervisorState
import structlog

logger = structlog.get_logger(__name__)

MAX_ITER = 8


async def chat_subgraph(state: SupervisorState, config: RunnableConfig) -> dict:
    from graph.llm_registry import get_llm
    from utils.mcp_connection_manager import mcp_manager
    from tools import AVAILABLE_TOOLS, get_tool

    enabled_tools   = state.get("enabled_tools", [])
    model           = state.get("model", "gemini-2.5-flash")
    skill_body      = state.get("skill_body", "")

    # Build tool list from user-enabled tools + MCP tools
    tools_to_bind = []
    tool_map = {}  # name → callable for execution
    for name in enabled_tools:
        if name in AVAILABLE_TOOLS:
            t = get_tool(name)
            if t:
                tools_to_bind.append(t)
                tool_map[name] = t

    mcp_tools = await mcp_manager.get_all_langchain_tools()
    tools_to_bind.extend(mcp_tools)
    for t in mcp_tools:
        tool_map[t.name] = t

    llm = get_llm(model)
    if tools_to_bind:
        llm = llm.bind_tools(tools_to_bind)

    # Inject skill into system message if triggered
    messages = list(state["messages"])
    if skill_body and messages and not isinstance(messages[0], SystemMessage):
        skill_msg = SystemMessage(content=f"## ACTIVE SKILL\n{skill_body}")
        messages = [skill_msg] + messages
    elif skill_body and messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(
            content=messages[0].content + f"\n\n## ACTIVE SKILL\n{skill_body}"
        )

    # ── ReAct loop (only if tools available) ────────────────────────────────
    for iteration in range(MAX_ITER):
        response = await llm.ainvoke(messages)
        messages.append(response)

        # If no tools or no tool calls, we're done
        if not tool_map or not (hasattr(response, "tool_calls") and response.tool_calls):
            break

        tool_outputs = []
        for tc in response.tool_calls:
            t = tool_map.get(tc["name"])
            if t:
                try:
                    result = await t.ainvoke(tc["args"])
                    logger.info(f"chat_subgraph.tool_call name={tc['name']} iter={iteration}")
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tc['name']}"
            tool_outputs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        messages.extend(tool_outputs)

    final = messages[-1]
    text = ""
    if isinstance(final.content, list):
        text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in final.content
        )
    else:
        text = str(final.content)

    return {"messages": [final], "final_response": text}
