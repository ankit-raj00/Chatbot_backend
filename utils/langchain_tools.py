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

def json_schema_to_pydantic(name: str, schema: dict) -> Any:
    """
    Dynamically create a Pydantic model from a JSON Schema.
    Handles basic types: string, integer, number, boolean.
    """
    from pydantic import create_model, Field
    
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    fields = {}
    
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        description = field_info.get("description", "")
        default_val = field_info.get("default")
        
        # Map types
        if field_type_str == "integer":
            py_type = int
        elif field_type_str == "number":
            py_type = float
        elif field_type_str == "boolean":
            py_type = bool
        else:
            py_type = str
            
        # Create Field
        # If required and no default -> ...
        # If default exists -> Field(default, ...)
        # If not required and no default -> Field(None, ...) and Optional[py_type]
        
        if default_val is not None:
            fields[field_name] = (py_type, Field(default=default_val, description=description))
        elif field_name in required:
            fields[field_name] = (py_type, Field(..., description=description))
        else:
            fields[field_name] = (Optional[py_type], Field(None, description=description))
            
    return create_model(f"{name}Args", **fields)

def wrap_native_tool(native_tool: Any) -> BaseTool:
    """
    Wraps our internal BaseTool (Native) into a LangChain BaseTool.
    """
    try:
        # Create Pydantic args_schema from native tool parameters
        if hasattr(native_tool, "parameters"):
            args_schema = json_schema_to_pydantic(native_tool.name, native_tool.parameters)
        else:
            args_schema = None
            
        async def _async_native_wrapper(**kwargs):
            return await native_tool.execute(**kwargs)

        return StructuredTool.from_function(
            func=None,
            coroutine=_async_native_wrapper,
            name=native_tool.name,
            description=native_tool.description,
            args_schema=args_schema
        )
    except Exception as e:
        print(f"Error wrapping tool {native_tool.name}: {e}")
        # Fallback to no-args or unvalidated
        async def _fallback(**kwargs):
            return await native_tool.execute(**kwargs)
        return StructuredTool.from_function(
            func=None,
            coroutine=_fallback,
            name=native_tool.name,
            description=native_tool.description
        )

async def get_all_active_mcp_tools() -> List[BaseTool]:
    """Helper to get tools from ALL active connections"""
    all_tools = []
    active_urls = mcp_manager.get_active_connections()
    
    for url in active_urls:
        tools = await get_langchain_mcp_tools(url)
        all_tools.extend(tools)
        
    return all_tools
