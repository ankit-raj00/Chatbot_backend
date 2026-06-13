"""
Native Tool Node: Executes Python-based tools
"""
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState
from tools import get_tool, AVAILABLE_TOOLS
from utils.hooks import run_pre_tool_hooks, run_post_tool_hooks, ToolTimer

async def native_tool_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes native tools detected in the last message.
    Handles user_id injection automatically.
    Executes all tools in parallel and uses the result cache for idempotent tools.
    """
    from utils.tool_result_cache import cached_invoke
    import inspect

    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    user_id = state.get("user_id")
    configuration = config.get("configurable", {})
    enabled_tool_names = configuration.get("enabled_tools", [])

    async def _execute_tool(tool_call: dict) -> ToolMessage:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        if tool_name not in AVAILABLE_TOOLS:
            # Not a native tool, skip
            return None
            
        if tool_name not in enabled_tool_names:
            return ToolMessage(
                content=f"Error: Tool '{tool_name}' is not enabled. The user has not granted permission in Settings.",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error"
            )

        tool_instance = get_tool(tool_name)
        if not tool_instance:
            return ToolMessage(
                content=f"Native tool '{tool_name}' not found.",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error"
            )

        func = tool_instance.func or tool_instance.coroutine
        sig = inspect.signature(func)
        
        exec_args = tool_args.copy()
        if "user_id" in sig.parameters and user_id:
            exec_args["user_id"] = user_id
             
        selected_files = state.get("selected_files")
        if "selected_files" in sig.parameters and selected_files is not None:
            exec_args["selected_files"] = selected_files

        # Hooks
        hook_result = await run_pre_tool_hooks(tool_name, tool_args, user_id or "")
        if hook_result and hook_result.get("deny"):
            return ToolMessage(
                content=f"Tool call blocked: {hook_result.get('reason', 'blocked by hook')}",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error"
            )

        if hook_result and hook_result.get("modify"):
            exec_args = hook_result.get("args", exec_args)

        # Execution wrapped in cache
        async def _run():
            if tool_instance.coroutine:
                return await tool_instance.ainvoke(exec_args)
            else:
                # If it's sync, wrap it in thread
                import asyncio
                return await asyncio.to_thread(tool_instance.invoke, exec_args)

        try:
            with ToolTimer() as timer:
                output = await cached_invoke(tool_name, exec_args, _run)
            await run_post_tool_hooks(tool_name, output, timer.elapsed_ms, user_id or "")
            return ToolMessage(content=str(output), name=tool_name, tool_call_id=tool_call_id, status="success")
        except Exception as e:
            return ToolMessage(content=f"Error executing {tool_name}: {str(e)}", name=tool_name, tool_call_id=tool_call_id, status="error")

    # Run all native tools in parallel
    tasks = [_execute_tool(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks)
    
    # Filter out Nones (MCP tools)
    final_results = [r for r in results if r is not None]
    
    return {"messages": final_results}

