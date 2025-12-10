from fastapi import HTTPException
from fastmcp import Client
from contextlib import AsyncExitStack

class MCPController:
    """Controller for MCP operations"""
    
    def __init__(self):
        self.local_mcp_client = None

    def _get_local_client(self):
        """Lazily initialize local client"""
        try:
            if not self.local_mcp_client:
                import pathlib
                mcp_server_path = pathlib.Path(__file__).parent.parent / "services" / "mcp_server.py"
                try:
                    self.local_mcp_client = Client(str(mcp_server_path))
                except Exception as e:
                    print(f"Warning: Failed to initialize local MCP client: {e}")
                    return None
            return self.local_mcp_client
        except Exception:
            return None
    
    async def connect_and_list_tools(self, mcp_server_url: str = None):
        """Connect to MCP server and list available tools"""
        tools_info = []
        
        try:
            async with AsyncExitStack() as stack:
                # Get local tools
                local_client = self._get_local_client()
                if local_client:
                    try:
                        await stack.enter_async_context(local_client)
                        local_tools = await local_client.list_tools()
                        for tool in local_tools:
                            tools_info.append({
                                "name": tool.name,
                                "description": tool.description,
                                "source": "Local"
                            })
                    except Exception as e:
                        print(f"Error listing local tools: {e}")

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
