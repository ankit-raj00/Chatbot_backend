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
    
    # Fetch MCP Tools dynamically
    mcp_tools = await get_all_active_mcp_tools()
    
    # Fetch Native Tools
    from tools import AVAILABLE_TOOLS
    # Wrap native tools for LangChain
    native_tools = []
    for tool_name, tool_instance in AVAILABLE_TOOLS.items():
        # Using a generic wrapper for our custom BaseTool to LangChain BaseTool
        # Check if already has .to_langchain_tool() or wrap manually
        # Ideally our native tools should be compatible or easy to wrap.
        # Let's import the wrapper logic or create a helper.
        # For fast fix, we define a wrapper helper here or in langchain_tools.
        from utils.langchain_tools import wrap_native_tool
        native_tools.append(wrap_native_tool(tool_instance))
    
    all_tools = mcp_tools + native_tools
    
    # Bind tools to the model
    if all_tools:
        llm = llm.bind_tools(all_tools)
        
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
