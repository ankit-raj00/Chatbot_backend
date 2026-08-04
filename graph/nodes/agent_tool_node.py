"""
agent_tool_node — executes ALL tool calls from agent_node in one place.

Unlike native_tool_node + mcp_tool_node (which split on AVAILABLE_TOOLS
membership), this node has direct access to the SAME tool instances
agent_node bound, via a lookup map rebuilt identically.
"""
import asyncio
import inspect
import json
from datetime import datetime
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
from utils.background_tasks import spawn
from core.database import turn_steps_collection
from config.model_config import ModelConfig
from pathlib import Path
import structlog

DEFAULT_MODEL = ModelConfig.DEFAULT_MODEL

logger = structlog.get_logger(__name__)


async def _log_turn_step(payload: dict) -> None:
    """Wraps the Motor insert in a real coroutine function — asyncio.create_task
    (used by spawn()) requires an actual coroutine object, and calling a Motor
    collection method directly can hand back a Future-like object instead,
    which create_task rejects with TypeError."""
    await turn_steps_collection.insert_one(payload)


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

# Result-aware stuck detection: the guards above only look at how many times a
# tool was CALLED. That misses the real signal — whether it's still returning
# NEW information. Repeating the exact same (tool, args) and getting the exact
# same result back, over and over, means zero new information is entering
# context; that's a much stronger "genuinely stuck" signal than a raw call
# count, and unlike the softer nudges above, this escalates hard immediately
# rather than waiting for a fixed threshold turn.
_STALE_RESULT_MAX = 2

# Periodic LLM-judge progress check (every N total tool calls this turn,
# across ALL tools) — a raw call-count threshold can't tell "genuinely
# iterating on a real task" apart from "stuck repeating," so periodically ask
# a cheap/fast model to look at the recent trajectory and classify it. This
# mirrors the reflection/progress-check node pattern used in LangGraph's own
# reference deep-research agent (an `evaluate_research`-style node reading
# recent (tool, args, result) tuples and setting a continue/stop signal).
_PROGRESS_CHECK_INTERVAL = 10
_PROGRESS_CHECK_WINDOW = 10


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


_SIMILAR_RESULT_PREFIX_LEN = 150
_SIMILAR_RESULT_WINDOW = 6
_SIMILAR_RESULT_MAX = 2


def _recent_similar_result_count(messages, name: str, content: str) -> int:
    """How many of the last _SIMILAR_RESULT_WINDOW ToolMessages for this SAME
    tool NAME (regardless of args) have a result whose leading chars match
    this one. Catches a pattern the exact-args stale check misses: re-reading
    the same file via slightly different commands (different flags/line
    ranges) that return overlapping content — proven live (conversation
    6a71a1bfa11cf10ed1232e44): 6 separate run_shell calls with different args
    all returned the same file's leading content, and none tripped the
    exact-args stale check since the args genuinely differed each time."""
    if not content or len(content) < 20:
        return 0
    prefix = content[:_SIMILAR_RESULT_PREFIX_LEN]
    same_name_results = [
        m.content for m in messages
        if isinstance(m, ToolMessage) and m.name == name and isinstance(m.content, str)
    ]
    recent = same_name_results[-_SIMILAR_RESULT_WINDOW:]
    return sum(1 for r in recent if r[:_SIMILAR_RESULT_PREFIX_LEN] == prefix)


def _total_tool_call_count(messages) -> int:
    """How many tool calls of ANY kind appear in the current turn."""
    n = 0
    for m in messages:
        n += len(getattr(m, "tool_calls", None) or [])
    return n


def _prior_results_for_call(messages, name: str, raw_args: dict) -> list[str]:
    """Content strings of every prior ToolMessage answering this exact
    (name, args) call this turn — used to detect a call returning the exact
    SAME result repeatedly (zero new information), not just being called
    repeatedly with possibly-changing results."""
    try:
        target = json.dumps(raw_args or {}, sort_keys=True, default=str)
    except Exception:
        target = str(raw_args)
    matching_ids = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") != name:
                continue
            try:
                if json.dumps(dict(tc.get("args") or {}), sort_keys=True, default=str) == target:
                    matching_ids.add(tc.get("id"))
            except Exception:
                pass
    if not matching_ids:
        return []
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in matching_ids:
            out.append(m.content if isinstance(m.content, str) else str(m.content))
    return out


def _recent_tool_trajectory(messages, window: int) -> list[dict]:
    """Compact (tool, args, result_snippet) view of the last `window` tool
    calls this turn, in order — fed to the progress-check judge instead of
    the full raw message history."""
    calls_by_id = {}
    order = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            cid = tc.get("id")
            calls_by_id[cid] = {"tool": tc.get("name"), "args": tc.get("args"), "result": None}
            order.append(cid)
    for m in messages:
        if isinstance(m, ToolMessage):
            cid = getattr(m, "tool_call_id", None)
            if cid in calls_by_id:
                content = m.content if isinstance(m.content, str) else str(m.content)
                calls_by_id[cid]["result"] = content[:300]
    trajectory = [calls_by_id[cid] for cid in order if cid in calls_by_id]
    return trajectory[-window:]


async def _progress_check(trajectory: list[dict], model_name: str) -> dict:
    """Ask a cheap/fast LLM call to classify the recent tool-call trajectory
    as genuine progress vs. an unproductive loop. Mirrors the periodic
    reflection/progress-check node pattern (e.g. LangGraph's own reference
    deep-research agent's evaluate_research node) rather than relying purely
    on a call-count threshold, which can't distinguish real iteration from
    stuck repetition. Never raises — on any failure, assumes "progress" so
    this safety net can't itself break a healthy turn."""
    try:
        from graph.llm_registry import get_llm
        from langchain_core.messages import HumanMessage

        steps_text = "\n".join(
            f"{i+1}. {t['tool']}({json.dumps(t['args'], default=str)}) -> {t['result']}"
            for i, t in enumerate(trajectory)
        )
        prompt = (
            "You are monitoring an AI agent's recent tool-call trajectory inside a "
            "single conversation turn. Decide if it is making genuine progress toward "
            "answering the user, or stuck in an unproductive loop (repeating "
            "essentially the same actions without new useful information).\n\n"
            f"Recent tool calls (oldest first):\n{steps_text}\n\n"
            'Respond with ONLY a compact JSON object: {"status": "progress" or "stuck", '
            '"reason": "<one short sentence>"}. No other text.'
        )
        llm = get_llm(model_name)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return {"status": "progress", "reason": "unparseable judge response"}
        parsed = json.loads(text[start:end + 1])
        if parsed.get("status") not in ("progress", "stuck"):
            return {"status": "progress", "reason": "invalid judge status"}
        return parsed
    except Exception as e:
        logger.warning("progress_check.failed", error=str(e))
        return {"status": "progress", "reason": f"judge error: {e}"}


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
    for t in (make_run_python_tool(user_id, conversation_id), make_run_shell_tool(user_id, conversation_id),
              make_edit_file_tool(user_id, conversation_id), make_analyze_image_tool(user_id, conversation_id),
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
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", "")
    turn_id = configurable.get("turn_id", "")
    model_name = configurable.get("model") or DEFAULT_MODEL

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

        # Hard block: list_skills has NO real parameters (confirmed schema:
        # {"properties": {}}) — the "reason" text seen in logs is the model
        # inventing an undeclared field, not a genuine argument. Because it's
        # a different string every call, the args-based checks above never
        # treat repeats as identical, even though the actual action (listing
        # a static, unchanging skill catalog for this user) is 100% identical
        # every time. There is never a legitimate reason to call it twice in
        # one turn — block outright after the first, rather than relying on
        # a nudge the model can (and, confirmed live, did) ignore.
        if name == "list_skills":
            prior_list_skills = _same_tool_call_count(state["messages"], "list_skills")
            if prior_list_skills >= 1:
                return ToolMessage(
                    content=(
                        "BLOCKED: you already called list_skills earlier this turn — it has no "
                        "parameters and returns the exact same static skill catalog every time, so "
                        "calling it again cannot give you new information. Use the skill list you "
                        "already have: call load_skill(skill_name) directly for the one that applies, "
                        "or if none apply, proceed without a skill manual."
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

            # Result-aware stuck detection: the exact same call returning the
            # exact same result, repeatedly, means zero new information is
            # entering context — escalate immediately regardless of the
            # softer count-based nudges above.
            if isinstance(tool_content, str):
                prior_results = _prior_results_for_call(state["messages"], name, tool_call.get("args") or {})
                stale = sum(1 for r in prior_results if r == tool_content)
                if stale >= _STALE_RESULT_MAX:
                    tool_content += (
                        f"\n\n🛑 STUCK: this exact call has now returned the IDENTICAL result "
                        f"{stale + 1} times — you are getting zero new information from repeating it. "
                        f"Do not call '{name}' with these arguments again. Either try a genuinely "
                        f"different approach, or give the user your final answer now, stating "
                        f"plainly what you found and what you couldn't resolve."
                    )
                else:
                    # Broader net: same TOOL (regardless of args) returning
                    # overlapping content — e.g. re-reading the same file via
                    # slightly different commands/flags. The exact-args check
                    # above can't catch this since the args genuinely differ.
                    similar = _recent_similar_result_count(state["messages"], name, tool_content)
                    if similar >= _SIMILAR_RESULT_MAX:
                        tool_content += (
                            f"\n\n🛑 STUCK: your last {similar + 1} calls to '{name}' returned overlapping/"
                            f"repeat content (even though the arguments differed each time) — you appear to "
                            f"be re-reading something you already have. Stop re-reading it. Use what you "
                            f"already learned, try a genuinely different action, or give your final answer now."
                        )

            # Incremental crash-durability log — fire-and-forget so a hard
            # server crash mid-turn loses at most the in-flight step, not the
            # whole turn (previously NOTHING was persisted until the entire
            # turn finished). TTL-expired (see main.py); the completed turn is
            # still saved in full to messages_collection as before.
            if turn_id:
                spawn(
                    _log_turn_step({
                        "turn_id": turn_id,
                        "conversation_id": configurable.get("thread_id", ""),
                        "user_id": user_id,
                        "step_index": _total_tool_call_count(state["messages"]),
                        "tool": name,
                        "args": args,
                        "status": "success",
                        "result_preview": (tool_content[:500] if isinstance(tool_content, str) else "[multimodal]"),
                        "created_at": datetime.utcnow(),
                    }),
                    name="turn_step_log",
                )

            return ToolMessage(content=tool_content, name=name, tool_call_id=call_id, status="success")
        except Exception as e:
            if turn_id:
                spawn(
                    _log_turn_step({
                        "turn_id": turn_id,
                        "conversation_id": configurable.get("thread_id", ""),
                        "user_id": user_id,
                        "step_index": _total_tool_call_count(state["messages"]),
                        "tool": name,
                        "args": args,
                        "status": "error",
                        "result_preview": str(e)[:500],
                        "created_at": datetime.utcnow(),
                    }),
                    name="turn_step_log",
                )
            return ToolMessage(content=f"Error executing {name}: {e}", name=name, tool_call_id=call_id, status="error")

    results = await asyncio.gather(*[_execute(tc) for tc in tool_calls])

    # Periodic LLM-judge progress check — every _PROGRESS_CHECK_INTERVAL total
    # tool calls this turn (across ALL tools), ask a cheap model to classify
    # the recent trajectory as progress vs. stuck, rather than relying purely
    # on the count-based guards above (which can't tell real iteration apart
    # from unproductive repetition).
    #
    # total_calls (state["messages"] already includes THIS round's triggering
    # AIMessage.tool_calls) is the cumulative count AFTER this batch. Checking
    # `total_calls % N == 0` is WRONG for parallel tool-calling: a round that
    # dispatches, say, 8 calls at once can jump the cumulative total straight
    # past a multiple of N (e.g. 55 -> 63, never landing on 60), silently
    # skipping the check for that round and every one after it that doesn't
    # happen to land exactly on a boundary either. Proven live: a 59-tool-call
    # stuck-loop run never triggered this check even once, specifically
    # because of this skip (conversation 6a71a1bfa11cf10ed1232e44). Fix:
    # detect whether a multiple of N falls ANYWHERE in (before, after].
    total_calls = _total_tool_call_count(state["messages"])
    total_before = total_calls - len(tool_calls)
    crossed_boundary = (total_before // _PROGRESS_CHECK_INTERVAL) != (total_calls // _PROGRESS_CHECK_INTERVAL)
    if total_calls > 0 and crossed_boundary:
        trajectory = _recent_tool_trajectory(state["messages"] + results, _PROGRESS_CHECK_WINDOW)
        verdict = await _progress_check(trajectory, model_name)
        if verdict.get("status") == "stuck":
            # Escalate on a REPEAT stuck verdict this turn — a model that
            # ignored the first soft nudge needs stronger, more absolute
            # language, not the same wording again.
            prior_stuck_checks = sum(
                1 for m in state["messages"]
                if isinstance(m, ToolMessage) and isinstance(m.content, str) and "PROGRESS CHECK" in m.content
            )
            if prior_stuck_checks == 0:
                directive = (
                    f"\n\n🛑 PROGRESS CHECK: after {total_calls} tool calls this turn, an automated review "
                    f"judged this trajectory as STUCK — {verdict.get('reason', 'no new progress detected')}. "
                    f"STOP making tool calls. Give the user your best final answer right now based on what "
                    f"you've already found, and clearly state what you were unable to resolve."
                )
            else:
                directive = (
                    f"\n\n🛑🛑 PROGRESS CHECK (repeat #{prior_stuck_checks + 1}): you were ALREADY told to stop "
                    f"and answer, and made MORE tool calls instead — {verdict.get('reason', 'still no new progress')}. "
                    f"This is your last warning before this turn is forcibly terminated. Do NOT call any more "
                    f"tools. Respond with your final answer THIS TURN, stating plainly what you completed and "
                    f"what you could not."
                )
            results = [
                ToolMessage(content=(r.content + directive) if isinstance(r.content, str) else r.content,
                            name=r.name, tool_call_id=r.tool_call_id, status=r.status)
                for r in results
            ]

    return {"messages": list(results)}
