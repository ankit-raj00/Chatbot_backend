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
        mcp_server_urls: List of MCP servers to use.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    conversation_id: Optional[str]
    enabled_tools: List[str]
    mcp_server_urls: List[str]
