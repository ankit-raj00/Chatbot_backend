from fastapi import APIRouter, Depends
from controllers.mcp_server_controller import MCPServerController
from models.mcp_server import MCPServerCreate, MCPServerUpdate
from core.middleware import get_current_user

router = APIRouter(prefix="/mcp-servers", tags=["MCP Servers"])

@router.get("")
async def get_servers(current_user: dict = Depends(get_current_user)):
    """Get all MCP servers for the current user"""
    return await MCPServerController.get_user_servers(str(current_user["_id"]))

@router.post("")
async def add_server(
    server: MCPServerCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a new MCP server"""
    return await MCPServerController.add_server(str(current_user["_id"]), server)

@router.put("/{server_id}")
async def update_server(
    server_id: str,
    updates: MCPServerUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update an MCP server"""
    return await MCPServerController.update_server(
        server_id,
        str(current_user["_id"]),
        updates
    )

@router.delete("/{server_id}")
async def delete_server(
    server_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an MCP server"""
    return await MCPServerController.delete_server(
        server_id,
        str(current_user["_id"])
    )

@router.post("/{server_id}/test")
async def test_connection(
    server_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Test MCP server connection and list tools"""
    return await MCPServerController.test_connection(
        server_id,
        str(current_user["_id"])
    )
