"""
Native Tool Node: Executes Python-based tools
"""
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState
from tools import get_tool, AVAILABLE_TOOLS

async def native_tool_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes native tools detected in the last message.
    Handles user_id injection automatically.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    results = []
    
    user_id = state.get("user_id")
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        # 0. Filter: Only handle Native Tools
        if tool_name not in AVAILABLE_TOOLS:
            # Skip MCP tools (handled by mcp_tool_node)
            continue
            
        # 1. Get Tool Instance
        tool_instance = get_tool(tool_name)
        
        if tool_instance:
            try:
                # 2. Inject user_id if supported/needed
                # We check the underlying function signature via the Pydantic tool wrapper
                # But our tools are now standard @tool functions.
                # Simplest way: Check if 'user_id' is in the arguments expected?
                # Actually, our new Pydantic models (like ListFoldersArgs) DO NOT have user_id.
                # The function *implementation* has it as an optional arg.
                # LangChain's `invoke` should handle passing extra kwargs if we pass them.
                
                # Let's inspect the function to see if it accepts user_id
                import inspect
                func = tool_instance.func or tool_instance.coroutine
                sig = inspect.signature(func)
                
                exec_args = tool_args.copy()
                if "user_id" in sig.parameters and user_id:
                     exec_args["user_id"] = user_id
                
                # 3. Execute
                if tool_instance.coroutine:
                    output = await tool_instance.ainvoke(exec_args)
                else:
                    output = tool_instance.invoke(exec_args)
                    
            except Exception as e:
                output = f"Error executing {tool_name}: {str(e)}"
        else:
            output = f"Native tool '{tool_name}' not found."
            
        # 4. Create ToolMessage
        results.append(ToolMessage(
            content=str(output),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="success" # Optional status flag for UI metadata?
        ))
        
    return {"messages": results}
