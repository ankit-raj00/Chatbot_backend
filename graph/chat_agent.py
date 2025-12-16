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
async def model_node(state: ChatState, config: dict):
    """
    The main model node that calls Gemini.
    """
    # Initialize Model (Gemini 1.5 Flash as requested)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )
    
    # Get enabled tools from config
    enabled_tools = config.get("configurable", {}).get("enabled_tools", [])
    
    # Fetch MCP Tools dynamically
    mcp_tools = await get_all_active_mcp_tools()
    
    # Fetch Native Tools
    from tools import AVAILABLE_TOOLS
    from utils.langchain_tools import wrap_native_tool
    
    native_tools = []
    for tool_name, tool_instance in AVAILABLE_TOOLS.items():
        native_tools.append(wrap_native_tool(tool_instance))
    
    all_tools = mcp_tools + native_tools
    
    # Filter tools
    if enabled_tools:
        # Note: We need to match by name. 
        # Native tools use their ID (e.g., 'roll_dice').
        # MCP tools use their displayed name? Or sanitized name?
        # Usually list contains tool names.
        final_tools = [t for t in all_tools if t.name in enabled_tools]
    else:
        # If no tools enabled, bind none? Or all? 
        # Requirement implies strict filtering. If list provided but empty -> No Tools.
        # If list is None -> maybe All? 
        # Controller passes [] if None. So [] means no tools.
        final_tools = []
        
        # Wait, if user didn't select ANY, usually means "Chat Only". 
        # But if enabled_tools is None (legacy), maybe allow all? 
        # Controller currently does: `enabled_tools or []`. So default is NO TOOLS.
        pass

    # Bind tools to the model
    if final_tools:
        llm = llm.bind_tools(final_tools)
        
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
