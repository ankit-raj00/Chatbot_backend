"""
Shared Definitions for LangGraph
"""
from typing import Annotated, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """
    Global State for the Chat Graph.
    
    Attributes:
        messages: History of LangChain messages (Human, AI, Tool).
        user_id: The ID of the current user (injected from API).
        conversation_id: The ID of the current conversation (MongoDB).
        enabled_tools: List of native tools enabled for this session.
        mcp_server_ids: MongoDB _ids of the MCP servers selected for this request.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    conversation_id: Optional[str]
    enabled_tools: List[str]
    mcp_server_ids: List[str]
    selected_files: Optional[List[str]]
    # Set by agent_tool_node when a turn hits its hard tool-call ceiling or a
    # repeat stuck-loop verdict — agent_node reads this and calls the LLM
    # WITHOUT binding any tools, so the model is structurally incapable of
    # making another tool call and must respond with plain text. Previously
    # the only enforcement was a stronger-worded prose warning ("this is your
    # last warning before this turn is forcibly terminated") that nothing
    # actually backed up — confirmed live: a benign prompt produced 29 tool
    # calls / ~30 LLM round-trips despite escalating warnings.
    force_final_answer: Optional[bool]
