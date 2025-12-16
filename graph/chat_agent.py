from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode, tools_condition
from utils.langchain_tools import get_all_active_mcp_tools
import os
import asyncio

# 1. State Definition
class ChatState(TypedDict):
    # 'messages' holds the full interaction including tool calls
    messages: Annotated[List[BaseMessage], add_messages]

# 2. Nodes
async def model_node(state: ChatState):
    """
    The main model node that calls Gemini.
    We fetch available tools dynamically here to ensure we always have the latest MCP tools.
    """
    # Initialize Model (Gemini 1.5 Flash as requested)
    # Note: Google API Key should be in env
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )
    
    # Fetch Tools dynamically
    tools = await get_all_active_mcp_tools()
    
    # Bind tools to the model
    if tools:
        llm = llm.bind_tools(tools)
        
    # Invoke
    response = await llm.ainvoke(state["messages"])
    
    return {"messages": [response]}

async def tool_node_wrapper(state: ChatState):
    """
    Wrapper for ToolNode to fetch tools dynamically.
    LangGraph's prebuilt ToolNode usually expects a static list.
    Since our MCP tools might change content (dynamic list), we recreate the node execution.
    Actually, for simplicity in Phase 1, let's fetch tools and use prebuilt ToolNode.
    """
    tools = await get_all_active_mcp_tools()
    print(f"Executing with {len(tools)} active tools")
    tool_node = ToolNode(tools)
    return await tool_node.ainvoke(state)


# 3. Graph Construction
builder = StateGraph(ChatState)

builder.add_node("model", model_node)
builder.add_node("tools", tool_node_wrapper)

builder.add_edge(START, "model")

# If model decided to call tools -> tools node
# Else -> END
builder.add_conditional_edges(
    "model",
    tools_condition
)

# After tools execute, go back to model
builder.add_edge("tools", "model")

# 4. Compiled Graph
chat_agent = builder.compile()
