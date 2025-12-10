from fastapi import APIRouter
from pydantic import BaseModel
from controllers.mcp_controller import MCPController
from utils.mcp_connection_manager import mcp_manager

router = APIRouter(prefix="/mcp", tags=["MCP"])

# Initialize MCP controller
mcp_controller = MCPController()

class ConnectRequest(BaseModel):
    mcp_server_url: str | None = None

class DisconnectRequest(BaseModel):
    mcp_server_url: str

class PreConnectRequest(BaseModel):
    mcp_server_url: str

@router.post("/connect")
async def connect_mcp(request: ConnectRequest):
    """Connect to MCP server and list available tools"""
    return await mcp_controller.connect_and_list_tools(request.mcp_server_url)

@router.post("/pre-connect")
async def pre_connect_mcp(request: PreConnectRequest):
    """Pre-connect to MCP server and fetch resources in background"""
    client = await mcp_manager.connect(request.mcp_server_url)
    if client:
        resources = mcp_manager.get_cached_resources(request.mcp_server_url)
        return {
            "success": True, 
            "message": f"Connected to {request.mcp_server_url}",
            "resources_count": len(resources)
        }
    else:
        return {
            "success": False,
            "message": f"Failed to connect to {request.mcp_server_url}"
        }

@router.post("/disconnect")
async def disconnect_mcp(request: DisconnectRequest):
    """Disconnect from MCP server and clear cache"""
    await mcp_manager.disconnect(request.mcp_server_url)
    return {"message": f"Disconnected from {request.mcp_server_url}", "success": True}

