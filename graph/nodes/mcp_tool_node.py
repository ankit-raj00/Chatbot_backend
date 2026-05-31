"""
MCP Tool Node: Executes tools via MCP Connection Manager
"""
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState
from tools import AVAILABLE_TOOLS
from utils.mcp_connection_manager import mcp_manager
from utils.hooks import run_pre_tool_hooks, run_post_tool_hooks, ToolTimer

async def mcp_tool_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes MCP tools detected in the last message.
    Delegates execution to the MCP Connection Manager.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        # 0. Filter: Only handle MCP Tools
        if tool_name in AVAILABLE_TOOLS:
            # Skip Native tools (handled by native_tool_node)
            continue
        
        try:
            # Pre-tool hook
            hook_result = await run_pre_tool_hooks(tool_name, tool_args, "")
            if hook_result and hook_result.get("deny"):
                output = f"Tool call blocked: {hook_result.get('reason', 'blocked by hook')}"
                results.append(ToolMessage(content=output, name=tool_name, tool_call_id=tool_call_id))
                continue

            if hook_result and hook_result.get("modify"):
                tool_args = hook_result.get("args", tool_args)

            with ToolTimer() as timer:
                result = await mcp_manager.call_tool_by_name(tool_name, tool_args)
            output = str(result)

            # Post-tool hook
            await run_post_tool_hooks(tool_name, output, timer.elapsed_ms, "")

        except Exception as e:
            output = f"Error executing MCP tool {tool_name}: {str(e)}"
            
        # Create ToolMessage
        results.append(ToolMessage(
            content=output,
            name=tool_name,
            tool_call_id=tool_call_id
        ))
        
    return {"messages": results}
