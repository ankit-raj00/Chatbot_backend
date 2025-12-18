"""
Tool to read content from an MCP resource URI
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any

class ReadResourceArgs(BaseModel):
    uri: str = Field(description="The URI of the MCP resource to read (e.g., 'memo://notes')")

@tool(args_schema=ReadResourceArgs)
async def read_mcp_resource(uri: str) -> Dict[str, Any]:
    """
    Hosted "Bridge" Tool: Enables the LLM to read MCP data.
    
    Architecture Note:
    - This is a "Native Tool" because it runs on the Host (Chatbot Backend), not on a remote MCP server.
    - It wraps the `mcp_manager.load_resource()` method to expose it as a callable Tool for the Agent.
    - Without this wrapper, the Agent knows the resources exist (from System Prompt) but has no "function" to read them.
    """
    from utils.mcp_connection_manager import mcp_manager
    
    try:
        content = await mcp_manager.load_resource(uri)
        return {"content": content}
    except Exception as e:
        return {"error": f"Failed to read resource {uri}: {str(e)}"}
