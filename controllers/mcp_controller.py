from fastapi import HTTPException
from utils.mcp_connection_manager import mcp_manager

class MCPController:
    """Controller for MCP operations"""
    
    def __init__(self):
        pass
    
    async def connect_and_list_tools(self, mcp_server_url: str = None):
        """Connect to MCP server and list available tools via Manager"""
        tools_info = []
        
        try:
            # 1. Connect if URL provided
            if mcp_server_url:
                success = await mcp_manager.connect(mcp_server_url)
                if not success:
                     raise HTTPException(
                        status_code=400,
                        detail=f"Failed to connect to {mcp_server_url}"
                    )
            
            # 2. Get All Tools (from all connected servers)
            # Note: This aggregates tools from all active connections
            langchain_tools = await mcp_manager.get_all_langchain_tools()
            
            for tool in langchain_tools:
                tools_info.append({
                    "name": tool.name,
                    "description": tool.description,
                    "source": "Remote" # Simplified: Manager abstracts source
                })
                        
            return {"status": "success", "tools": tools_info}
            
        except Exception as e:
            print(f"Error in MCP Connect/List: {e}")
            if mcp_server_url:
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
            return {"status": "error", "detail": str(e)}
