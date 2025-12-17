"""
MCP Tool Node: Executes tools via MCP Connection Manager
"""
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState
from utils.mcp_connection_manager import mcp_manager

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
        
        try:
            # Execute via manager
            # This handles finding the tool across multiple servers
            result = await mcp_manager.call_tool_by_name(tool_name, tool_args)
            output = str(result)
            
            # Note: MCP result might be a list of Content objects, or text.
            # We convert to string for the LLM.
            
        except Exception as e:
            output = f"Error executing MCP tool {tool_name}: {str(e)}"
            
        # Create ToolMessage
        results.append(ToolMessage(
            content=output,
            name=tool_name,
            tool_call_id=tool_call_id
        ))
        
    return {"messages": results}
