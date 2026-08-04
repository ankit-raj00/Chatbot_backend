"""
agent_node — THE single agent. Replaces graph/supervisor.py's
intent_classifier_node + all 7 subgraphs.
"""
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, ToolMessage
from graph.nodes.common import ChatState  # reuse existing TypedDict, see 2.6
from tools.utilities.run_python import make_run_python_tool
from tools.utilities.run_shell import make_run_shell_tool
from tools.utilities.edit_file import make_edit_file_tool
from tools.utilities.analyze_image import make_analyze_image_tool
from tools.utilities.skill_tools import list_skills, make_load_skill_tool
from tools.utilities.read_file_natively import make_read_file_natively_tool
from tools import AVAILABLE_TOOLS, get_tool
from utils.mcp_connection_manager import mcp_manager

AGENT_SYSTEM_PROMPT = """You are AgentX — a single, capable AI agent with
direct access to a Python sandbox, a shell, your knowledge base, and a
library of skill manuals.

## Your tools
- run_python: execute Python scripts (data analysis, PDF/DOCX/PPTX/XLSX
  generation, computation). Pass `filename` (e.g. "work/build_report.py")
  whenever the script is more than a one-line check — this SAVES it as a real
  file instead of discarding it after the run, so if it needs a fix you can
  use edit_file on it and re-run via run_shell, instead of resending the whole
  script through run_python again. NOT for searching the internet — see
  internet_search below; do not write requests/urllib code to query a
  search engine yourself.
- run_shell: run shell commands in your sandboxed workspace (inspect files,
  run a script you saved with run_python's `filename`, check output).
- edit_file: make a targeted fix to an EXISTING file (old_string -> new_string).
  PREFER THIS over resending a whole file's content whenever you're fixing/
  tweaking something you (or the user) already created — a script you saved
  with run_python(filename=...), a LaTeX file, a config, etc. Re-emitting an
  entire file just to change a few lines wastes tokens and is more error-prone
  than a targeted edit. Only regenerate a whole file from scratch when the
  changes are pervasive enough that a full rewrite is genuinely simpler.
- Reading an existing file you need to inspect: prefer ONE read over many
  small paginated ones. `run_shell("cat file.py")` (or `grep -n pattern
  file.py` when you only need specific lines/definitions) gets you the whole
  picture in a single call. Do NOT read a file in a sequence of separate
  run_python calls each printing a 50-100 line slice — that burns many tool
  calls and turns for information one call already gives you.
- analyze_image: your VISION tool. To SEE any image — one the user uploaded
  (uploads/…) OR one you generated yourself (outputs/…, work/…) — call
  analyze_image(sandbox_path, query). It returns a text answer describing what's
  in the image. This is the ONLY way to look at an image; you cannot see image
  pixels otherwise. Use it to answer questions about uploaded pictures AND to
  visually verify your own output — e.g. after generating a chart or rendering a
  PDF page to PNG, analyze_image it to confirm it looks right before finishing.
- read_file_natively: minor helper for locating an uploaded file. For images,
  use analyze_image instead; for data/code files, read them with run_python/
  run_shell.
- list_skills / load_skill: your skill library. Skills are step-by-step
  manuals for specific tasks (creating PDFs, analyzing data, reviewing code,
  generating diagrams, etc).
- search_knowledge_base / read_document_page: search the user's KNOWLEDGE BASE
  — documents they ingested through the document-upload/knowledge-base feature.
  ⚠️ These documents are NOT sandbox files. They exist only as searchable text
  in the knowledge base; they will NEVER show up in ls/uploads/ and cannot be
  opened with run_shell/run_python. If the user says "my document", "my
  uploaded document", "the file I uploaded", "according to my document" (etc.)
  and you have NOT seen a file attached to THIS specific message, that almost
  always means their knowledge base — call search_knowledge_base FIRST, before
  touching the sandbox at all.
- internet_search: search the live web for anything NOT in the knowledge base
  — current events, external docs, prices, anything time-sensitive or you're
  unsure of. Cite the source URL it returns when you use a result.
  🚫 HARD RULE: internet_search is ALWAYS your first and only move for
  internet search. NEVER use run_python/run_shell to fetch a search engine, a
  news RSS feed, or any "let me write a script to query DuckDuckGo/Bing/
  Yahoo/Google News" approach — this is strictly forbidden, not just
  discouraged. It is slower, more fragile (redirect links, rate limits, HTML
  parsing), and pointlessly reimplements a tool you already have. If
  internet_search itself returns nothing useful, say so — do not fall back
  to hand-rolled scraping.
- Google Drive, weather, dice, time, and any connected MCP tools.

## Two different kinds of "uploaded file" — do not confuse them
1. A file attached directly to this chat message (multimodal upload) → lives
   in the sandbox's uploads/ folder → inspect with run_shell/run_python/
   analyze_image/read_file_natively.
2. A document the user ingested via the knowledge-base upload feature (at any
   earlier point, possibly a different session) → lives ONLY as vectors in the
   knowledge base → searchable ONLY with search_knowledge_base. It is not a
   file, `ls` will never find it, and trying to `cd`/`find`/`cat` your way to
   it will just fail repeatedly.
If a message doesn't include a fresh attachment, assume case 2 and go straight
to search_knowledge_base. If you're not sure, try search_knowledge_base first
anyway — it's cheap and immediately tells you whether there's a match.

## If you're not finding something — stop and switch strategy
Check the workspace snapshot FIRST — it already tells you what sandbox files
exist, so most "does this file exist" questions need zero tool calls at all.
If you still can't locate something after that: if 2 sandbox-exploration
calls (ls, find, cd, cat, etc.) in a row haven't located what you're looking
for, DO NOT keep guessing more paths/flags. Stop and do one of: (a) call
search_knowledge_base if this could be a knowledge-base document, (b) tell
the user plainly what you looked for and that you couldn't find it, and ask
them to clarify or re-attach it. Repeatedly retrying `ls`/`cd` with slightly
different arguments is not progress — recognize that pattern in your own
recent tool calls and break out of it.

## How to work
Think step by step. For non-trivial tasks:
1. If a skill might apply, call load_skill first to see the recommended
   approach — don't guess at library usage when a manual exists.
2. For anything beyond a trivial one-off check, SAVE your script with
   run_python(filename=...) and run it via run_shell rather than only passing
   inline code — this is what makes it editable afterward. Once a file exists
   (a saved script, a generated document, LaTeX source, etc.), use edit_file
   for subsequent fixes rather than rewriting it wholesale.
3. Chain as many tool calls as needed across the SAME turn — e.g. load a
   skill, run a script, load a second skill, run another script — before
   giving your final answer.
4. When a file is created, tell the user its name. DO NOT generate markdown download links yourself (e.g. `[file](url)`); the system will automatically surface a clickable file card for the user.

Always verify your work (check script output, fix errors, retry) before
declaring success.

CRITICAL INSTRUCTION FOR MULTI-STEP TASKS:
If the user asks you to perform multiple operations (e.g. create a file, THEN analyze it, THEN write a report), you MUST NEVER STOP after completing just the first step. You MUST immediately invoke the next necessary tool in your very next response. DO NOT ask for permission to continue. Keep generating tool calls sequentially until the ENTIRE multi-step request is complete.

## Your sandbox
You have a persistent personal sandbox with three folders:
- uploads/  — files attached directly to a chat message (read-only in
  practice; inspect freely). This does NOT include knowledge-base documents —
  see the section above.
- outputs/  — save files here that the user should be able to download
- work/     — your scratch space for intermediate files, extracted archives, etc.

A "Current workspace snapshot" is appended below, listing every file that
actually exists right now, freshly regenerated before each of your turns.
⚠️ TRUST IT — do NOT run `ls`/`find`/`cat ~/.bash_history` just to check what
files exist; that information is already given to you for free below. Only
use run_shell/run_python to look INSIDE a file's content, or to do real work.
If you're resuming a task from earlier in the conversation, check the
snapshot first — it tells you immediately whether your previous output
(e.g. outputs/report.pdf) still exists, without any exploration calls.

Your run_python and run_shell tools execute in this sandbox's root directory.
- To inspect an upload: run_shell("file uploads/whatever") or
  run_python("import pandas as pd; df = pd.read_csv('uploads/data.csv')")
- To extract an archive: run_shell("unzip uploads/report.zip -d work/report")
- To deliver a result: save to outputs/, e.g. plt.savefig('outputs/chart.png')
- Before calling a tool, briefly say what you're about to do and why (one short
  sentence) — this is shown to the user live as you work.
- Your run_python uses your own isolated Python environment (separate
  virtualenv) — pip installs are private to you and persist across messages
  in the same session.
"""


from config.model_config import ModelConfig
DEFAULT_MODEL = ModelConfig.DEFAULT_MODEL

# Max messages (excluding the always-kept system message) retained per LLM call.
# Bounds context growth across a long conversation AND a long multi-tool ReAct turn.
# Raised from 40: old ToolMessage payloads are now COMPACTED (see
# _compact_stale_tool_messages) before this count-based backstop applies, so
# the same message budget holds far more actual turns without losing the
# model's own reasoning/narration messages, which is what caused a real
# production stuck-loop (see _compact_stale_tool_messages docstring).
MAX_TURN_MESSAGES = 100

# How many of the MOST RECENT messages stay fully intact (untouched) when
# compacting. Older ToolMessages beyond this get their content replaced with
# a short placeholder — see _compact_stale_tool_messages.
_RECENT_UNCOMPACTED_MESSAGES = 20

# Only compact ToolMessage content longer than this many characters — short
# results are cheap to keep as-is and compacting them saves nothing.
_COMPACT_CONTENT_THRESHOLD = 400

async def agent_node(state: ChatState, config: RunnableConfig) -> dict:
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "anonymous")
    conversation_id = configuration.get("thread_id", "")
    enabled_tool_names = configuration.get("enabled_tools", [])
    mcp_server_ids = configuration.get("mcp_server_ids", [])
    model_name = configuration.get("model", DEFAULT_MODEL)

    # ── Build the flat tool list ──────────────────────────────────────────
    tools = []

    # 1. Always-on sandbox tools (the core of "free will")
    tools.append(make_run_python_tool(user_id))
    tools.append(make_run_shell_tool(user_id))
    tools.append(make_edit_file_tool(user_id))
    tools.append(make_analyze_image_tool(user_id))
    tools.append(make_read_file_natively_tool(user_id, conversation_id))

    # 2. Skill tools (always on — cheap, and the agent needs to discover them)
    tools.append(list_skills)
    tools.append(make_load_skill_tool(user_id))

    # 3. User-enabled native tools (RAG search, Drive, weather, etc.)
    for name in enabled_tool_names:
        if name in AVAILABLE_TOOLS:
            t = get_tool(name)
            if t:
                tools.append(t)

    # 4. MCP tools — scoped to THIS user and THIS request's selected servers
    #    (never every server ever connected by any user in the process).
    mcp_tools = await mcp_manager.get_tools_for_servers(user_id, mcp_server_ids)
    tools.extend(mcp_tools)

    # ── Get LLM, bind tools ─────────────────────────────────────────────
    from graph.llm_registry import get_llm
    llm = get_llm(model_name).bind_tools(tools)

    # ── Ensure exactly one system message = base prompt built by chat_service
    #    (PromptBuilder) merged with the agent's operating instructions ────────
    messages = _with_system_prompt(list(state["messages"]))

    # ── Inject a freshly-scanned workspace snapshot EVERY call (not just once
    #    per turn) — this runs on every agent_node invocation in the ReAct
    #    loop, so as soon as a tool creates a file, the very next LLM call
    #    already sees it, without the model needing to ls/find to check. This
    #    directly targets the observed failure mode where an agent resuming a
    #    task (or losing track mid-turn) burned dozens of calls rediscovering
    #    file state it could have been simply told.
    messages = _inject_workspace_snapshot(messages, user_id)

    # ── Trim to a bounded window so long histories / long ReAct turns can't
    #    blow the context. Always keep the system message + most recent turns ──
    messages = _trim(messages)

    # Wrap the LLM call in the circuit breaker so a gateway outage fails fast
    # instead of hanging every concurrent request through retry backoff.
    from utils.circuit_breaker import gemini_breaker
    response = await gemini_breaker.call(llm.ainvoke, messages)
    return {"messages": [response]}


def _as_text(content) -> str:
    """Normalize message content (str OR list of multimodal parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        return "\n".join(parts)
    return str(content or "")


def _with_system_prompt(messages: list) -> list:
    """Guarantee a single leading SystemMessage that includes AGENT_SYSTEM_PROMPT.

    chat_service prepends its own PromptBuilder SystemMessage; we augment that one
    with the agent operating instructions. If none exists, we add ours. Content is
    normalized to str first so appending never breaks on multimodal (list) content.
    """
    if messages and isinstance(messages[0], SystemMessage):
        base = _as_text(messages[0].content)
        if AGENT_SYSTEM_PROMPT not in base:
            messages[0] = SystemMessage(content=f"{base}\n\n{AGENT_SYSTEM_PROMPT}")
        return messages
    return [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages


_SNAPSHOT_MAX_FILES_PER_DIR = 25
_SNAPSHOT_START = "<<<WORKSPACE_SNAPSHOT>>>"
_SNAPSHOT_END = "<<<END_WORKSPACE_SNAPSHOT>>>"


def _workspace_snapshot(user_id: str) -> str:
    """Fast filesystem scan of uploads/work/outputs — ground truth on what
    actually exists, computed fresh (no caching, no staleness risk) every
    time this is called. Deliberately cheap: no file content is read, only
    names/sizes/mtimes, so this is safe to run on every single LLM call."""
    from utils.workspace import workspace_for
    ws = workspace_for(user_id)
    lines = []
    for sub in ("uploads", "work", "outputs"):
        d = ws / sub
        files = [f for f in d.rglob("*") if f.is_file()] if d.exists() else []
        if not files:
            lines.append(f"{sub}/: (empty)")
            continue
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        lines.append(f"{sub}/ ({len(files)} file{'s' if len(files) != 1 else ''}, newest first):")
        for f in files[:_SNAPSHOT_MAX_FILES_PER_DIR]:
            rel = f.relative_to(ws)
            lines.append(f"  - {rel} ({f.stat().st_size:,} bytes)")
        if len(files) > _SNAPSHOT_MAX_FILES_PER_DIR:
            lines.append(f"  ... and {len(files) - _SNAPSHOT_MAX_FILES_PER_DIR} more (older)")
    return "\n".join(lines)


def _inject_workspace_snapshot(messages: list, user_id: str) -> list:
    """Replace (or add) the workspace-snapshot block in the leading
    SystemMessage with a freshly-computed one. Uses delimiter markers so this
    can safely re-run on every agent_node call within a turn without
    duplicating or stacking stale copies."""
    try:
        snapshot = _workspace_snapshot(user_id)
    except Exception:
        return messages  # never let a filesystem hiccup break a turn
    block = f"{_SNAPSHOT_START}\n## Current workspace snapshot (auto-generated — trust it, don't re-check with ls/find)\n{snapshot}\n{_SNAPSHOT_END}"

    if not messages or not isinstance(messages[0], SystemMessage):
        return messages
    base = _as_text(messages[0].content)
    if _SNAPSHOT_START in base and _SNAPSHOT_END in base:
        pre = base[:base.index(_SNAPSHOT_START)]
        post = base[base.index(_SNAPSHOT_END) + len(_SNAPSHOT_END):]
        new_base = f"{pre}{block}{post}"
    else:
        new_base = f"{base}\n\n{block}"
    messages[0] = SystemMessage(content=new_base)
    return messages


def _compact_stale_tool_messages(messages: list) -> list:
    """Compact old tool-call payloads instead of hard-dropping whole messages
    (Anthropic's context-engineering guidance: prefer compaction over
    truncation). Proven necessary live: a 30+ tool-call production turn
    exceeded the old flat 40-message trim, silently deleting the messages
    showing the agent had already built a PDF and fixed layout issues — the
    model then "forgot" its own progress and fell into an unproductive
    re-exploration loop (repeated ls/list_skills) until it hit the recursion
    limit.

    This keeps every message (and the tool_call/ToolMessage pairing the API
    requires) but replaces old, already-consumed ToolMessage CONTENT with a
    short placeholder — so the model's own reasoning/narration (AIMessage
    text — "I already did X") survives regardless of turn length, while the
    heavy raw tool output (file listings, shell output, search results) gets
    shed once it's no longer recent enough to plausibly matter.
    """
    if len(messages) <= _RECENT_UNCOMPACTED_MESSAGES:
        return messages
    cutoff = len(messages) - _RECENT_UNCOMPACTED_MESSAGES
    out = []
    for i, m in enumerate(messages):
        if (i < cutoff and isinstance(m, ToolMessage)
                and isinstance(m.content, str) and len(m.content) > _COMPACT_CONTENT_THRESHOLD):
            preview = m.content[:150].replace("\n", " ")
            out.append(ToolMessage(
                content=(f"[{m.name} result omitted — {len(m.content)} chars, already used earlier "
                         f"this turn. Preview: {preview}...]"),
                name=m.name, tool_call_id=m.tool_call_id, status=getattr(m, "status", "success"),
            ))
        else:
            out.append(m)
    return out


def _trim(messages: list) -> list:
    """Compact stale tool payloads, then keep the system message + the last
    MAX_TURN_MESSAGES messages as a hard backstop, without splitting a tool
    call from its ToolMessage result."""
    messages = _compact_stale_tool_messages(messages)
    from langchain_core.messages import trim_messages
    try:
        return trim_messages(
            messages,
            token_counter=len,              # count messages, not tokens (cheap, model-agnostic)
            max_tokens=MAX_TURN_MESSAGES,
            strategy="last",
            include_system=True,
            start_on="human",
            allow_partial=False,
        )
    except Exception:
        # Never let trimming break a turn — fall back to the full list.
        return messages
