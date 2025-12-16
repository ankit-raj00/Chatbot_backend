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

from langchain_core.runnables import RunnableConfig

# 2. Nodes
async def model_node(state: ChatState, config: RunnableConfig):
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
    # Logic: 
    # 1. Native Tools: Must be in 'enabled_tools' list.
    # 2. MCP Tools: Always allowed if connected (authentication handled by server connection).
    #    We identify native tools by checking if their name is in AVAILABLE_TOOLS.
    
    native_tool_names = set(AVAILABLE_TOOLS.keys())
    
    final_tools = []
    for tool in all_tools:
        if tool.name in native_tool_names:
            # It's a native tool, check if enabled
            if enabled_tools is None or tool.name in enabled_tools:
                final_tools.append(tool)
        else:
            # It's an MCP tool (or unknown), allow it
            final_tools.append(tool)

    # Bind tools to the model
    if final_tools:
        llm = llm.bind_tools(final_tools)
        
    # Invoke
    response = await llm.ainvoke(state["messages"])
    
    return {"messages": [response]}

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

async def tool_node_wrapper(state: ChatState, config: RunnableConfig):
    """
    Custom ToolNode that executes tools SEQUENTIALLY.
    This is required because the frontend assumes 'Start -> End' order (LIFO) for tool steps.
    Parallel execution (default ToolNode) causes race conditions in the UI.
    Also injects 'user_id' into Native Tools.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Get user_id from config
    user_id = config.get("configurable", {}).get("user_id")
    
    # 1. Prepare Tools Map
    mcp_tools = await get_all_active_mcp_tools()
    
    from tools import AVAILABLE_TOOLS
    from utils.langchain_tools import wrap_native_tool
    
    native_tools = []
    # Key = wrapped name, Value = native instance (to check properties)
    native_instances = {} 
    
    for tool_instance in AVAILABLE_TOOLS.values():
        wrapped = wrap_native_tool(tool_instance)
        native_tools.append(wrapped)
        native_instances[wrapped.name] = tool_instance
        
    all_tools = mcp_tools + native_tools
    tool_map = {t.name: t for t in all_tools}
    
    results = []
    
    # 2. Iterate and Execute Sequentially
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            selected_tool = tool_map.get(tool_name)
            
            if selected_tool:
                try:
                    # Execute
                    if tool_name in native_instances:
                        # NATIVE TOOL EXECUTION
                        # execute directly to allow hidden user_id injection without schema validation issues
                        native_tool = native_instances[tool_name]
                        
                        # Prepare args
                        exec_args = tool_args.copy()
                        if user_id:
                             exec_args["user_id"] = user_id
                             
                        output = await native_tool.execute(**exec_args)
                    else:
                        # MCP / STANDARD TOOL EXECUTION
                        # Use standard LangChain invocation
                        output = await selected_tool.ainvoke(tool_args, config=config)

                except Exception as e:
                    output = f"Error executing {tool_name}: {str(e)}"
            else:
                output = f"Tool {tool_name} not found."
            
            # Create ToolMessage
            # Ensure output is string for ToolMessage content
            content_str = str(output)
            
            results.append(ToolMessage(
                content=content_str,
                tool_call_id=tool_call_id,
                name=tool_name
            ))
            
    return {"messages": results}


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
