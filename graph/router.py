"""
Router: Decides next step (Native vs MCP vs End)
"""
from typing import List, Literal
from graph.nodes.common import ChatState
from tools import AVAILABLE_TOOLS
from langgraph.graph import END

def route_tools(state: ChatState) -> Literal["native_tool_node", "mcp_tool_node", "__end__", list]:
    """
    Router function to determine the next node based on tool calls.
    Supports parallel execution of Native and MCP nodes.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If no tool calls, stop
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    
    tool_calls = last_message.tool_calls
    has_native = False
    has_mcp = False
    
    # Check each call
    for call in tool_calls:
        name = call["name"]
        if name in AVAILABLE_TOOLS:
            has_native = True
        else:
            # Assume any unknown tool is MCP
            has_mcp = True
            
    # Routing decision
    if has_native and has_mcp:
        return ["native_tool_node", "mcp_tool_node"]
    elif has_native:
        return "native_tool_node"
    elif has_mcp:
        return "mcp_tool_node"
    else:
        # Should not happen if tool_calls exist, but fail-safe
        return END
