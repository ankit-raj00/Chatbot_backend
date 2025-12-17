"""
Graph Builder: Assembles the Chat Graph
"""
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.nodes.common import ChatState
from graph.nodes.setup_node import setup_node
from graph.nodes.native_tool_node import native_tool_node
from graph.nodes.mcp_tool_node import mcp_tool_node
from graph.router import route_tools
from tools import AVAILABLE_TOOLS
from utils.mcp_connection_manager import mcp_manager

async def chat_model_node(state: ChatState, config: RunnableConfig):
    """
    Core LLM Node.
    Binds tools dynamically based on configuration.
    """
    # 1. Get Configuration
    configuration = config.get("configurable", {})
    enabled_tool_names = configuration.get("enabled_tools", [])
    model_name = configuration.get("model", "gemini-2.0-flash-exp")
    
    # 2. Collect Tools
    tools_to_bind = []
    
    # Native Tools
    for name in enabled_tool_names:
        if name in AVAILABLE_TOOLS:
            # Import on demand to avoid circular deps if any
            from tools import get_tool
            # Because we wrapped them with @tool, get_tool returns the structured tool
            t = get_tool(name)
            if t:
                tools_to_bind.append(t)
                
    # MCP Tools (always bind all available from active connections)
    # Note: efficient because get_all_active_mcp_tools uses cached sessions
    mcp_tools = await mcp_manager.get_all_langchain_tools()
    tools_to_bind.extend(mcp_tools)
    
    # 3. Initialize Model
    # We use a fresh instance to ensure clean tool binding each turn
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        max_tokens=None,
        max_retries=2,
    )
    
    if tools_to_bind:
        llm = llm.bind_tools(tools_to_bind)
        
    # 4. Invoke
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    """Constructs the executable LangGraph"""
    builder = StateGraph(ChatState)
    
    # Add Nodes
    builder.add_node("setup_node", setup_node)
    builder.add_node("chat_model", chat_model_node)
    builder.add_node("native_tool_node", native_tool_node)
    builder.add_node("mcp_tool_node", mcp_tool_node)
    
    # Add Edges
    builder.add_edge(START, "setup_node")
    builder.add_edge("setup_node", "chat_model")
    
    # Conditional Edge (Router)
    builder.add_conditional_edges(
        "chat_model",
        route_tools,
        {
            "native_tool_node": "native_tool_node",
            "mcp_tool_node": "mcp_tool_node",
            "native_and_mcp": ["native_tool_node", "mcp_tool_node"], # Parallel map
            "__end__": END
        }
    )
    
    # Return from tools back to model
    builder.add_edge("native_tool_node", "chat_model")
    builder.add_edge("mcp_tool_node", "chat_model")
    
    return builder.compile()

# Singleton for easy import
chat_graph = build_graph()
