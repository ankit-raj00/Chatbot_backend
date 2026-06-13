"""
Supervisor Graph — the central router for AgentX v2.

Routes every incoming message to the right specialist subgraph:
  chat      → general conversation, Q&A, follow-up
  shell     → file system, commands, scripts, code execution
  document  → PDF, DOCX, PPTX, Excel generation
  vision    → image analysis, screenshots, multimodal
  code      → code generation, review, debugging
  rag       → document search, knowledge base
  data      → CSV/Excel analysis, statistics

Intent classification uses gemini-2.5-flash-lite (~50ms, very cheap).
Checkpointing uses Upstash Redis (langgraph-checkpoint-redis).
Falls back to MemorySaver if Redis unavailable.
"""

import os
import json
import logging
from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import structlog
logger = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")

INTENT_SYSTEM = """You are an intent classifier for a multi-agent AI system.
Given the user's message, classify which specialist agent should handle it.

Agents:
- chat:     general Q&A, conversation, writing, summarization, translation
- shell:    file system operations, bash commands, running scripts, exploring repos
- document: creating PDF, Word (.docx), PowerPoint (.pptx), Excel files
- vision:   analyzing images, screenshots, diagrams, charts
- code:     writing code, debugging, code review, refactoring
- rag:      searching the user's knowledge base or uploaded documents
- data:     analyzing CSV/Excel data, statistics, charts from data

Respond with ONLY the agent name in lowercase. No punctuation, no explanation."""

INTENT_FEW_SHOT = [
    ("create a PDF report on Q3 sales", "document"),
    ("run this python script", "shell"),
    ("what does this image show?", "vision"),
    ("search my documents for the contract clause", "rag"),
    ("analyze this CSV file for trends", "data"),
    ("write a FastAPI endpoint for user auth", "code"),
    ("what is machine learning?", "chat"),
    ("make me a PowerPoint presentation", "document"),
]

VALID_AGENTS = {"chat", "shell", "document", "vision", "code", "rag", "data"}


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    conversation_id: str
    agent: str            # Chosen agent name
    model: str
    enabled_tools: list[str]
    selected_files: Optional[list[str]]
    skill_body: str       # Injected skill content for subgraph
    skill_name: str       # Name of the injected skill
    final_response: str   # Accumulated text response


async def intent_classifier_node(state: SupervisorState) -> dict:
    """Classify the latest user message → pick an agent. Uses flash-lite for speed."""
    from graph.llm_registry import get_llm

    messages = state["messages"]
    if not messages:
        return {"agent": "chat"}

    # Get last human message
    last_human = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            if isinstance(m.content, list):
                last_human = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in m.content
                )
            else:
                last_human = str(m.content)
            break

    if not last_human:
        return {"agent": "chat"}

    # Build few-shot prompt
    few_shot = "\n".join(f'User: "{u}" → {a}' for u, a in INTENT_FEW_SHOT)
    user_prompt = f'{few_shot}\nUser: "{last_human}" →'

    try:
        llm  = get_llm("gemini-2.5-flash-lite")
        resp = await llm.ainvoke([
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        agent = resp.content.strip().lower()
        if agent not in VALID_AGENTS:
            agent = "chat"
    except Exception as e:
        logger.warning(f"intent_classifier failed: {e} — falling back to chat")
        agent = "chat"

    # Skill injection: check if a relevant skill exists for this agent + message
    skill_body = ""
    skill_name = ""
    try:
        from skills.skill_loader import get_relevant_skill_for_message
        user_id = state.get("user_id", "")
        skill_res = await get_relevant_skill_for_message(last_human, user_id, agent)
        if skill_res:
            skill_body, skill_name = skill_res
    except Exception:
        pass

    logger.info(f"supervisor.routed agent={agent} skill={'yes' if skill_body else 'no'}")
    return {"agent": agent, "skill_body": skill_body, "skill_name": skill_name}


def _route_to_agent(state: SupervisorState) -> str:
    """Conditional edge: supervisor → specialist subgraph."""
    return state.get("agent", "chat")


def _build_supervisor():
    from graph.subgraphs.chat_subgraph     import chat_subgraph
    from graph.subgraphs.shell_subgraph    import shell_subgraph
    from graph.subgraphs.document_subgraph import document_subgraph
    from graph.subgraphs.vision_subgraph   import vision_subgraph
    from graph.subgraphs.code_subgraph     import code_subgraph
    from graph.subgraphs.rag_subgraph      import rag_subgraph
    from graph.subgraphs.data_subgraph     import data_subgraph

    builder = StateGraph(SupervisorState)

    # Central intent node
    builder.add_node("intent_classifier", intent_classifier_node)

    # Specialist subgraph nodes
    builder.add_node("chat",     chat_subgraph)
    builder.add_node("shell",    shell_subgraph)
    builder.add_node("document", document_subgraph)
    builder.add_node("vision",   vision_subgraph)
    builder.add_node("code",     code_subgraph)
    builder.add_node("rag",      rag_subgraph)
    builder.add_node("data",     data_subgraph)

    # Edges
    builder.add_edge(START, "intent_classifier")
    builder.add_conditional_edges(
        "intent_classifier",
        _route_to_agent,
        {a: a for a in VALID_AGENTS},
    )
    for a in VALID_AGENTS:
        builder.add_edge(a, END)

    return builder


async def _init_checkpointer():
    """Try Redis checkpointer (Upstash). Fall back to MemorySaver."""
    if REDIS_URL:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            # AsyncRedisSaver.from_conn_string is an async context manager;
            # we enter it and keep the reference open for the app lifetime.
            cm = AsyncRedisSaver.from_conn_string(REDIS_URL)
            checkpointer = await cm.__aenter__()
            logger.info("supervisor.checkpointer=redis")
            return checkpointer, cm   # return both so we can __aexit__ on shutdown
        except Exception as e:
            logger.warning(f"Redis checkpointer failed ({e}) — using MemorySaver")
    from langgraph.checkpoint.memory import MemorySaver
    logger.info("supervisor.checkpointer=memory")
    return MemorySaver(), None


_supervisor_instance = None
_supervisor_checkpointer = None
_supervisor_checkpointer_cm = None   # context manager handle for clean shutdown


async def get_supervisor():
    """Return compiled supervisor graph singleton (async — initialises Redis on first call)."""
    global _supervisor_instance, _supervisor_checkpointer, _supervisor_checkpointer_cm
    if _supervisor_instance is None:
        _supervisor_checkpointer, _supervisor_checkpointer_cm = await _init_checkpointer()
        builder = _build_supervisor()
        _supervisor_instance = builder.compile(checkpointer=_supervisor_checkpointer)
        logger.info("supervisor.compiled")
    return _supervisor_instance


async def close_supervisor():
    """Graceful shutdown — close Redis connection pool."""
    global _supervisor_checkpointer_cm
    if _supervisor_checkpointer_cm is not None:
        try:
            await _supervisor_checkpointer_cm.__aexit__(None, None, None)
            logger.info("supervisor.checkpointer.closed")
        except Exception as e:
            logger.warning(f"supervisor.checkpointer close error: {e}")
        _supervisor_checkpointer_cm = None
