"""
Tool to read content from an MCP resource URI
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class ReadResourceArgs(BaseModel):
    uri: str = Field(description="The URI of the MCP resource to read (e.g., 'memo://notes')")

@tool(args_schema=ReadResourceArgs)
async def read_mcp_resource(uri: str, user_id: str = "", mcp_server_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Hosted "Bridge" Tool: Enables the LLM to read MCP data.

    Architecture Note:
    - This is a "Native Tool" because it runs on the Host (Chatbot Backend), not on a remote MCP server.
    - It wraps `mcp_manager.load_resource()` to expose it as a callable Tool for the Agent.
    - Without this wrapper, the Agent knows the resources exist (from System Prompt) but has no "function" to read them.
    - `user_id`/`mcp_server_ids` are injected automatically by agent_tool_node,
      scoping the search to this user's currently-selected servers only.
    """
    from utils.mcp_connection_manager import mcp_manager

    try:
        content = await mcp_manager.load_resource(user_id, mcp_server_ids or [], uri)
        return {"content": content}
    except Exception as e:
        return {"error": f"Failed to read resource {uri}: {str(e)}"}
