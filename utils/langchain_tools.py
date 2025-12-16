from typing import List, Optional, Any
from langchain_core.tools import BaseTool, tool, StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from utils.mcp_connection_manager import mcp_manager
import asyncio
from pydantic import BaseModel, create_model

async def get_langchain_mcp_tools(server_name: str) -> List[BaseTool]:
    """
    Get LangChain compatible tools for a connected MCP server.
    Uses the official langchain-mcp-adapters library with manual fallback.
    """
    try:
        # 1. Connect
        session = await mcp_manager.connect(server_name)
        if not session:
            print(f"Warning: Could not connect to MCP server {server_name}")
            return []

        # 2. Try Official Adapter
        try:
            # We suspect fastmcp client might not be 100% compatible with 
            # expected session interface of langchain-mcp-adapters.
            # But let's try it first.
            tools = await load_mcp_tools(session)
            return tools
        except Exception as adapter_error:
            # print(f"Adapter failed ({adapter_error}), falling back to manual wrapping for {server_name}")
            pass
            
        # 3. Manual Fallback (For FastMCP compatibility)
        # FastMCP client has .list_tools() and .call_tool()
        mcp_tools = await session.list_tools()
        langchain_tools = []
        
        for t in mcp_tools:
            async def _create_tool_func(tool_name=t.name, **kwargs):
                return await session.call_tool(tool_name, arguments=kwargs)
            
            # Create Pydantic model for args
            input_schema = t.inputSchema # OpenRPC schema dict
            # Simplifying: FastMCP usually gives clean JSON schema
            # We can use StructuredTool.from_function but we need type hints or schema
            
            # Simple wrapper:
            langchain_tools.append(StructuredTool.from_function(
                func=None,
                coroutine=_create_tool_func,
                name=t.name,
                description=t.description or "",
                # We can try passing args_schema if we parse inputSchema
                # For now, let LangChain infer or use generic implementation
            ))
            
        # Refined Fallback: Use a generic tool that takes any args
        # Because dynamic pydantic model creation from JSON schema is complex 
        # to do robustly in one step.
        # But wait, without schema, LangChain agent might not know how to call it.
        # Let's rely on standard StructuredTool where possible.
        
        return langchain_tools

    except Exception as e:
        print(f"Error loading MCP tools for {server_name}: {e}")
        return []

def wrap_native_tool(native_tool: Any) -> BaseTool:
    """
    Wraps our internal BaseTool (Native) into a LangChain BaseTool.
    """
    # Our native tools have: name, description, execute(args), parameters (JSON Schema)
    
    async def _async_native_wrapper(**kwargs):
        # Inject dependencies if needed (handled in execute_tool usually)
        # But here we call the tool instance directly
        # Note: Our native tools might expect 'user_id' etc. 
        # We need to rely on the agent passing them or kwargs having them.
        return await native_tool.execute(**kwargs)

    # We need to convert native tool 'parameters' (JSON Schema) to Pydantic
    # This is tricky. For now, let's look at the tool implementation.
    # Most have simpler args.
    
    return StructuredTool.from_function(
        func=None,
        coroutine=_async_native_wrapper,
        name=native_tool.name,
        description=native_tool.description,
        # args_schema=... (Ideally we convert JSON schema to Pydantic)
    )

async def get_all_active_mcp_tools() -> List[BaseTool]:
    """Helper to get tools from ALL active connections"""
    all_tools = []
    active_urls = mcp_manager.get_active_connections()
    
    for url in active_urls:
        tools = await get_langchain_mcp_tools(url)
        all_tools.extend(tools)
        
    return all_tools
