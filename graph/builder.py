"""
Graph Builder v3 — single ReAct agent.
Replaces graph/supervisor.py and graph/builder.py (v2 flat-chat version).

NOTE: the graph is compiled WITHOUT a persistent checkpointer. Conversation
history is loaded fresh from MongoDB every turn (services/history_service.py) and
passed as the graph input, so a cross-invocation checkpointer would be a *second*
source of truth for the same messages. With `add_messages`, the reconstructed
history (new message IDs each turn) would be APPENDED to the persisted state
rather than merged — making state and token cost grow super-linearly and
duplicating the system prompt. Dropping the checkpointer removes that entire
class of bug. Intra-turn ReAct state (agent_node <-> agent_tool_node) still works
because state persists for the duration of a single astream_events() call.
"""
import structlog
from langgraph.graph import StateGraph, START, END

from graph.nodes.common import ChatState
from graph.nodes.agent_node import agent_node
from graph.nodes.agent_tool_node import agent_tool_node

logger = structlog.get_logger(__name__)


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


_graph_instance = None


async def get_agent_graph():
    """Return the compiled (stateless-across-turns) agent graph, built once."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = _build_graph().compile()
        logger.info("agent_graph.compiled checkpointer=none")
    return _graph_instance


async def close_agent_graph():
    """No-op — no checkpointer connection to close. Kept for lifespan compatibility."""
    return None
