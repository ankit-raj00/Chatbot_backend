"""
Graph Builder v3 — single ReAct agent.
Replaces graph/supervisor.py and graph/builder.py (v2 flat-chat version).
"""
import os
import structlog
from langgraph.graph import StateGraph, START, END

from graph.nodes.common import ChatState
from graph.nodes.agent_node import agent_node
from graph.nodes.agent_tool_node import agent_tool_node

logger = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")


def _route_after_agent(state: ChatState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "agent_tool_node"
    return END


def _build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("agent_node", agent_node)
    builder.add_node("agent_tool_node", agent_tool_node)

    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges("agent_node", _route_after_agent, ["agent_tool_node", END])
    builder.add_edge("agent_tool_node", "agent_node")

    return builder


async def _init_checkpointer():
    """Same Redis-or-MemorySaver pattern as old supervisor._init_checkpointer."""
    if REDIS_URL:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            cm = AsyncRedisSaver.from_conn_string(REDIS_URL)
            checkpointer = await cm.__aenter__()
            logger.info("agent_graph.checkpointer=redis")
            return checkpointer, cm
        except Exception as e:
            logger.warning(f"Redis checkpointer failed ({e}) — using MemorySaver")
    from langgraph.checkpoint.memory import MemorySaver
    logger.info("agent_graph.checkpointer=memory")
    return MemorySaver(), None


_graph_instance = None
_checkpointer = None
_checkpointer_cm = None


async def get_agent_graph():
    global _graph_instance, _checkpointer, _checkpointer_cm
    if _graph_instance is None:
        _checkpointer, _checkpointer_cm = await _init_checkpointer()
        _graph_instance = _build_graph().compile(checkpointer=_checkpointer)
        logger.info("agent_graph.compiled")
    return _graph_instance


async def close_agent_graph():
    global _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"agent_graph checkpointer close error: {e}")
        _checkpointer_cm = None
