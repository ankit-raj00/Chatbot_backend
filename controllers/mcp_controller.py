from fastapi import HTTPException
from fastmcp import Client
from contextlib import AsyncExitStack

class MCPController:
    """Controller for MCP operations"""
    
    def __init__(self):
        import pathlib
        import os
        mcp_server_path = pathlib.Path(__file__).parent.parent / "mcp_server.py"
        self.local_mcp_client = Client(str(mcp_server_path))
    
    async def connect_and_list_tools(self, mcp_server_url: str = None):
        """Connect to MCP server and list available tools"""
        tools_info = []
        
        try:
            async with AsyncExitStack() as stack:
                # Get local tools
                await stack.enter_async_context(self.local_mcp_client)
                local_tools = await self.local_mcp_client.list_tools()
                for tool in local_tools:
                    tools_info.append({
                        "name": tool.name,
                        "description": tool.description,
                        "source": "Local"
                    })

                # Get remote tools if URL provided
                if mcp_server_url:
                    remote_client = Client(mcp_server_url)
                    await stack.enter_async_context(remote_client)
                    remote_tools = await remote_client.list_tools()
                    for tool in remote_tools:
                        tools_info.append({
                            "name": tool.name,
                            "description": tool.description,
                            "source": "Remote"
                        })
                        
            return {"status": "success", "tools": tools_info}
            
        except Exception as e:
            print(f"Error connecting to MCP: {e}")
            if mcp_server_url:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to connect to remote server: {str(e)}"
                )
            return {"status": "error", "detail": str(e)}
