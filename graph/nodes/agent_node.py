"""
agent_node — THE single agent. Replaces graph/supervisor.py's
intent_classifier_node + all 7 subgraphs.
"""
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from graph.nodes.common import ChatState  # reuse existing TypedDict, see 2.6
from tools.utilities.run_python import make_run_python_tool
from tools.utilities.run_shell import make_run_shell_tool
from tools.utilities.skill_tools import list_skills, make_load_skill_tool
from tools import AVAILABLE_TOOLS, get_tool
from utils.mcp_connection_manager import mcp_manager

AGENT_SYSTEM_PROMPT = """You are AgentX — a single, capable AI agent with
direct access to a Python sandbox, a shell, your knowledge base, and a
library of skill manuals.

## Your tools
- run_python: execute Python scripts (data analysis, PDF/DOCX/PPTX/XLSX
  generation, computation).
- run_shell: run shell commands in your sandboxed workspace (inspect files,
  run scripts you wrote, check output).
- list_skills / load_skill: your skill library. Skills are step-by-step
  manuals for specific tasks (creating PDFs, analyzing data, reviewing code,
  generating diagrams, etc).
- search_knowledge_base / read_document_page: search the user's uploaded
  documents.
- Google Drive, weather, dice, time, and any connected MCP tools.

## How to work
Think step by step. For non-trivial tasks:
1. If a skill might apply, call load_skill first to see the recommended
   approach — don't guess at library usage when a manual exists.
2. Use run_python / run_shell to do the actual work, checking output as
   you go.
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
- uploads/  — files the user has sent you (read-only in practice; inspect freely)
- outputs/  — save files here that the user should be able to download
- work/     — your scratch space for intermediate files, extracted archives, etc.

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

async def agent_node(state: ChatState, config: RunnableConfig) -> dict:
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "anonymous")
    enabled_tool_names = configuration.get("enabled_tools", [])
    model_name = configuration.get("model", DEFAULT_MODEL)

    # ── Build the flat tool list ──────────────────────────────────────────
    tools = []

    # 1. Always-on sandbox tools (the core of "free will")
    tools.append(make_run_python_tool(user_id))
    tools.append(make_run_shell_tool(user_id))

    # 2. Skill tools (always on — cheap, and the agent needs to discover them)
    tools.append(list_skills)
    tools.append(make_load_skill_tool(user_id))

    # 3. User-enabled native tools (RAG search, Drive, weather, etc.)
    for name in enabled_tool_names:
        if name in AVAILABLE_TOOLS:
            t = get_tool(name)
            if t:
                tools.append(t)

    # 4. Connected MCP tools
    mcp_tools = await mcp_manager.get_all_langchain_tools()
    tools.extend(mcp_tools)

    # ── Get LLM, bind tools ─────────────────────────────────────────────
    from graph.llm_registry import get_llm
    llm = get_llm(model_name).bind_tools(tools)

    # ── Inject system prompt (once, if not already present) ──────────────
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages
    elif AGENT_SYSTEM_PROMPT not in messages[0].content:
        messages[0] = SystemMessage(content=messages[0].content + "\n\n" + AGENT_SYSTEM_PROMPT)

    response = await llm.ainvoke(messages)
    return {"messages": [response]}
