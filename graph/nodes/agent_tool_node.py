"""
agent_tool_node — executes ALL tool calls from agent_node in one place.

Unlike native_tool_node + mcp_tool_node (which split on AVAILABLE_TOOLS
membership), this node has direct access to the SAME tool instances
agent_node bound, via a lookup map rebuilt identically.
"""
import asyncio
import inspect
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState
from tools.utilities.run_python import make_run_python_tool
from tools.utilities.run_shell import make_run_shell_tool
from tools.utilities.edit_file import make_edit_file_tool
from tools.utilities.analyze_image import make_analyze_image_tool
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


# Loop guardrail: the ReAct agent can get stuck repeating the exact same tool
# call (e.g. `ls` over and over) without making progress — especially recovering
# from a flaky/restarted LLM gateway. With recursion_limit=150 that can grind for
# ~75 rounds. If the same (tool, args) has already run this many times in the
# current turn, we refuse to run it again and feed back a corrective message so
# the model breaks out instead of looping to the hard limit.
_LOOP_MAX_IDENTICAL = 3

# Broader guardrail: the identical-args guard above misses a common failure
# mode — many DIFFERENT calls to the same exploration tool (ls, then ls -la,
# then find, then cd ..) that never converge because the model is hunting for
# something that isn't actually a sandbox file (e.g. a knowledge-base document
# it should have searched with search_knowledge_base instead). Past this many
# calls to the same tool in one turn, we append a nudge to the result so the
# model sees the pattern and can course-correct, without blocking execution
# (unlike the hard identical-args block — these ARE different commands, some
# of which may still be legitimate).
_LOOP_NUDGE_TOOLS = {"run_shell", "run_python", "search_knowledge_base", "list_skills"}
_LOOP_NUDGE_THRESHOLD = 5

# Even broader guardrail: a model avoiding the per-tool threshold above by
# HOPPING between different exploration tools (search_knowledge_base, then
# run_shell, then list_skills, then back to search_knowledge_base...) never
# trips any single-tool counter, yet still burns through the graph's
# recursion_limit and dies on a hard "step limit reached" error instead of
# ever answering. Proven live: a user with zero ingested documents alternated
# search_knowledge_base/run_shell/list_skills for 270+s and never converged.
# Count ALL calls to ANY tool in this combined set, regardless of which one,
# and nudge once the total crosses the threshold.
_LOOP_NUDGE_COMBINED_THRESHOLD = 8


def _same_tool_call_count(messages, name: str) -> int:
    """How many times ANY call to this tool name appears in the current turn."""
    n = 0
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == name:
                n += 1
    return n


def _combined_tool_call_count(messages, names: set) -> int:
    """How many calls to ANY tool in `names` appear in the current turn."""
    n = 0
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") in names:
                n += 1
    return n


def _identical_call_count(messages, name: str, raw_args: dict) -> int:
    """How many times this exact (name, args) tool call already appears across
    the AI messages of the current turn (history from prior turns is stored as
    plain text and carries no tool_calls, so this naturally scopes to this turn)."""
    try:
        target = json.dumps(raw_args or {}, sort_keys=True, default=str)
    except Exception:
        target = str(raw_args)
    n = 0
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == name:
                try:
                    if json.dumps(dict(tc.get("args") or {}), sort_keys=True, default=str) == target:
                        n += 1
                except Exception:
                    pass
    return n


async def _build_tool_map(state: ChatState, config: RunnableConfig) -> dict:
    """Rebuild the same tool list agent_node bound, keyed by tool.name."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "anonymous")
    conversation_id = configuration.get("thread_id", "")
    enabled_tool_names = configuration.get("enabled_tools", [])
    mcp_server_ids = configuration.get("mcp_server_ids", [])
    selected_files = state.get("selected_files")

    tool_map = {}

    # Always-on sandbox tools — must exactly mirror agent_node.py's tool list
    for t in (make_run_python_tool(user_id), make_run_shell_tool(user_id),
              make_edit_file_tool(user_id), make_analyze_image_tool(user_id),
              list_skills, make_load_skill_tool(user_id)):
        tool_map[t.name] = (t, {"user_id": user_id, "selected_files": selected_files})

    # read_file_natively needs conversation_id to look up the Gemini URI
    rfn_tool = make_read_file_natively_tool(user_id, conversation_id)
    tool_map[rfn_tool.name] = (rfn_tool, {})

    for name in enabled_tool_names:
        if name in AVAILABLE_TOOLS:
            t = get_tool(name)
            if t:
                tool_map[t.name] = (t, {
                    "user_id": user_id,
                    "selected_files": selected_files,
                    "mcp_server_ids": mcp_server_ids,
                })

    for t in await mcp_manager.get_tools_for_servers(user_id, mcp_server_ids):
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

        # Loop guardrail — refuse an exact call that's already been made too many
        # times this turn, and tell the model to stop repeating and converge.
        prior = _identical_call_count(state["messages"], name, tool_call.get("args") or {})
        if prior > _LOOP_MAX_IDENTICAL:
            return ToolMessage(
                content=(
                    f"LOOP GUARD: you have already called '{name}' with these exact arguments "
                    f"{prior} times and keep getting the same result — this is not making progress. "
                    f"STOP repeating this call. Take a genuinely different action, or if you are "
                    f"stuck, give the user your best final answer now and clearly explain what you "
                    f"found and what is blocking you."
                ),
                name=name, tool_call_id=call_id, status="error",
            )

        entry = tool_map.get(name)
        if not entry:
            return ToolMessage(
                content=f"Error: tool '{name}' not found.",
                name=name, tool_call_id=call_id, status="error",
            )
        tool_obj, extra = entry

        # Inject user_id / selected_files if the tool signature wants them.
        # SECURITY: user_id is always FORCED from the authenticated config and
        # must never be overridable by the model — otherwise a hallucinated or
        # injected user_id in a tool call could read another tenant's data.
        _FORCED_KEYS = {"user_id"}
        func = tool_obj.func or tool_obj.coroutine
        if func:
            sig = inspect.signature(func)
            for key, val in extra.items():
                if key in sig.parameters and val is not None:
                    if key in _FORCED_KEYS or key not in args:
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
                
            # If the tool returned a list (multimodal content parts, e.g. from
            # read_file_natively), pass it directly so the LLM receives the
            # actual image/file data.  For everything else, stringify as before.
            tool_content = output if isinstance(output, list) else str(output)

            # Nudge (not block) if an exploration tool is being called a lot
            # this turn without switching strategy — see _LOOP_NUDGE_TOOLS above.
            if name in _LOOP_NUDGE_TOOLS and isinstance(tool_content, str):
                same_count = _same_tool_call_count(state["messages"], name) + 1
                if same_count == _LOOP_NUDGE_THRESHOLD:
                    if name in ("run_shell", "run_python"):
                        tool_content += (
                            f"\n\n⚠️ NOTICE: you've called '{name}' {same_count} times this turn without "
                            f"a clear result. If you're looking for content the user referred to as 'my "
                            f"document'/'uploaded document', that is very likely a KNOWLEDGE-BASE document, "
                            f"not a sandbox file — call search_knowledge_base instead of continuing to "
                            f"explore the sandbox. Otherwise, stop guessing paths and tell the user what "
                            f"you looked for and that you couldn't find it."
                        )
                    else:
                        tool_content += (
                            f"\n\n⚠️ NOTICE: you've called '{name}' {same_count} times this turn. If you "
                            f"already have enough information to answer the user's question, do so now "
                            f"instead of searching further. If after this many attempts you still haven't "
                            f"found what you're looking for, it most likely doesn't exist — tell the user "
                            f"plainly rather than continuing to retry variations."
                        )
                else:
                    combined_count = _combined_tool_call_count(state["messages"], _LOOP_NUDGE_TOOLS) + 1
                    if combined_count == _LOOP_NUDGE_COMBINED_THRESHOLD:
                        tool_content += (
                            f"\n\n⚠️ NOTICE: you've made {combined_count} exploration/search tool calls "
                            f"this turn (across search_knowledge_base, run_shell, run_python, list_skills "
                            f"combined) without reaching an answer. STOP exploring now — give the user "
                            f"your best final answer immediately, clearly stating what you found (or that "
                            f"you found nothing, e.g. no matching documents in their knowledge base). "
                            f"Do not make another tool call before answering."
                        )
            return ToolMessage(content=tool_content, name=name, tool_call_id=call_id, status="success")
        except Exception as e:
            return ToolMessage(content=f"Error executing {name}: {e}", name=name, tool_call_id=call_id, status="error")

    results = await asyncio.gather(*[_execute(tc) for tc in tool_calls])
    return {"messages": list(results)}
