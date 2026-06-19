"""
agent_tool_node — executes ALL tool calls from agent_node in one place.

Unlike native_tool_node + mcp_tool_node (which split on AVAILABLE_TOOLS
membership), this node has direct access to the SAME tool instances
agent_node bound, via a lookup map rebuilt identically.
"""
import asyncio
import inspect
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState
from tools.utilities.run_python import make_run_python_tool
from tools.utilities.run_shell import make_run_shell_tool
from tools.utilities.skill_tools import list_skills, make_load_skill_tool
from tools.utilities.read_file_natively import make_read_file_natively_tool
from tools import AVAILABLE_TOOLS, get_tool
from utils.mcp_connection_manager import mcp_manager
from utils.hooks import run_pre_tool_hooks, run_post_tool_hooks, ToolTimer
from utils.tool_result_cache import cached_invoke
from pathlib import Path


def _infer_project_type(cwd_path: Path) -> str:
    if (cwd_path / "package.json").exists():
        return "node"
    if (cwd_path / "requirements.txt").exists() or (cwd_path / ".venv").exists():
        return "python"
    return "generic"


async def _build_tool_map(state: ChatState, config: RunnableConfig) -> dict:
    """Rebuild the same tool list agent_node bound, keyed by tool.name."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "anonymous")
    conversation_id = configuration.get("thread_id", "")
    enabled_tool_names = configuration.get("enabled_tools", [])
    selected_files = state.get("selected_files")

    tool_map = {}

    # Always-on sandbox tools — must exactly mirror agent_node.py's tool list
    for t in (make_run_python_tool(user_id), make_run_shell_tool(user_id),
              list_skills, make_load_skill_tool(user_id)):
        tool_map[t.name] = (t, {"user_id": user_id, "selected_files": selected_files})

    # read_file_natively needs conversation_id to look up the Gemini URI
    rfn_tool = make_read_file_natively_tool(user_id, conversation_id)
    tool_map[rfn_tool.name] = (rfn_tool, {})

    for name in enabled_tool_names:
        if name in AVAILABLE_TOOLS:
            t = get_tool(name)
            if t:
                tool_map[t.name] = (t, {"user_id": user_id, "selected_files": selected_files})

    for t in await mcp_manager.get_all_langchain_tools():
        tool_map[t.name] = (t, {})

    return tool_map


async def agent_tool_node(state: ChatState, config: RunnableConfig) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    user_id = config.get("configurable", {}).get("user_id", "")

    tool_map = await _build_tool_map(state, config)

    async def _execute(tool_call: dict) -> ToolMessage:
        name = tool_call["name"]
        args = dict(tool_call["args"])
        call_id = tool_call["id"]

        entry = tool_map.get(name)
        if not entry:
            return ToolMessage(
                content=f"Error: tool '{name}' not found.",
                name=name, tool_call_id=call_id, status="error",
            )
        tool_obj, extra = entry

        # Inject user_id / selected_files if the tool signature wants them
        func = tool_obj.func or tool_obj.coroutine
        if func:
            sig = inspect.signature(func)
            for key, val in extra.items():
                if key in sig.parameters and val is not None and key not in args:
                    args[key] = val

        # Hooks (same as native_tool_node)
        hook_result = await run_pre_tool_hooks(name, args, user_id)
        if hook_result and hook_result.get("deny"):
            return ToolMessage(
                content=f"Tool call blocked: {hook_result.get('reason','blocked by hook')}",
                name=name, tool_call_id=call_id, status="error",
            )
        if hook_result and hook_result.get("modify"):
            args = hook_result.get("args", args)

        async def _run():
            if tool_obj.coroutine:
                return await tool_obj.ainvoke(args)
            return await asyncio.to_thread(tool_obj.invoke, args)

        try:
            with ToolTimer() as timer:
                output = await cached_invoke(name, args, _run)
            await run_post_tool_hooks(name, output, timer.elapsed_ms, user_id)
            
            if name in ("run_python", "run_shell"):
                from utils.workspace import workspace_for
                from utils.workspace_cleanup import touch_last_active
                ws = workspace_for(user_id)
                touch_last_active(user_id, _infer_project_type(ws))
                
            return ToolMessage(content=str(output), name=name, tool_call_id=call_id, status="success")
        except Exception as e:
            return ToolMessage(content=f"Error executing {name}: {e}", name=name, tool_call_id=call_id, status="error")

    results = await asyncio.gather(*[_execute(tc) for tc in tool_calls])
    return {"messages": list(results)}
