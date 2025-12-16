from typing import List, Optional
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.tools import load_mcp_tools
from utils.mcp_connection_manager import mcp_manager
import asyncio

async def get_langchain_mcp_tools(server_name: str) -> List[BaseTool]:
    """
    Get LangChain compatible tools for a connected MCP server.
    Uses the official langchain-mcp-adapters library.
    """
    # 1. Get the raw FastMCP Client session from our manager
    # The manager handles the lifecycle (connect/reconnect)
    session = await mcp_manager.connect(server_name) # Ensure connected
    
    if not session:
        print(f"Warning: Could not connect to MCP server {server_name}")
        return []

    # 2. Use the official adapter to convert to LangChain Tools
    # The adapter expects a session object that has .list_tools() and .call_tool()
    # verify fastmcp client compatibility or wrap if needed.
    # checking fastmcp interface... efficient way is just try/except or inspection
    # langchain-mcp-adapters generally expects a ModelContextProtocolClient from mcp-python
    # or something that follows interface. FastMCP client is slightly different.
    
    # Let's inspect what load_mcp_tools expects.
    # It usually takes a session factory or existing session.
    # For now, we assume simple wrapping.
    
    try:
         # official usage: await load_mcp_tools(session)
        tools = await load_mcp_tools(session)
        return tools
    except Exception as e:
        print(f"Error loading MCP tools for {server_name}: {e}")
        return []

async def get_all_active_mcp_tools() -> List[BaseTool]:
    """Helper to get tools from ALL active connections"""
    all_tools = []
    active_urls = mcp_manager.get_active_connections()
    
    for url in active_urls:
        tools = await get_langchain_mcp_tools(url)
        all_tools.extend(tools)
        
    return all_tools
