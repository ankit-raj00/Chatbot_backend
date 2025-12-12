from fastapi import HTTPException, status
from core.database import mcp_servers_collection
from models.mcp_server import MCPServerCreate, MCPServerUpdate
from datetime import datetime
from bson import ObjectId
try:
    from fastmcp import Client
except ImportError:
    Client = None

class MCPServerController:
    """Controller for MCP server operations"""
    
    @staticmethod
    async def get_user_servers(user_id: str):
        """Get all MCP servers for a user"""
        try:
            servers_cursor = mcp_servers_collection.find({
                "$or": [
                    {"user_id": user_id},
                    {"is_local": True}
                ]
            }).sort("created_at", -1)
            servers_list = await servers_cursor.to_list(length=100)
            
            # Convert ObjectId to string
            for server in servers_list:
                server["_id"] = str(server["_id"])
                if "created_at" in server and isinstance(server["created_at"], datetime):
                    server["created_at"] = server["created_at"].isoformat()
                if "updated_at" in server and isinstance(server["updated_at"], datetime):
                    server["updated_at"] = server["updated_at"].isoformat()
                elif "updated_at" not in server:
                    # If updated_at is missing (e.g. local server), use created_at or current time
                    server["updated_at"] = server.get("created_at") or datetime.now().isoformat()
            
            return servers_list
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def add_server(user_id: str, server: MCPServerCreate):
        """Add a new MCP server"""
        try:
            new_server = {
                "user_id": user_id,
                "name": server.name,
                "url": server.url,
                "is_active": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            result = await mcp_servers_collection.insert_one(new_server)
            new_server["_id"] = str(result.inserted_id)
            new_server["created_at"] = new_server["created_at"].isoformat()
            new_server["updated_at"] = new_server["updated_at"].isoformat()
            return new_server
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def update_server(server_id: str, user_id: str, updates: MCPServerUpdate):
        """Update an MCP server"""
        try:
            # Verify server exists
            server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id)})
            
            if not server:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Server not found"
                )
            
            # Prevent updating local server
            if server.get("is_local"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot update local system server"
                )
            
            # Verify ownership
            if server.get("user_id") != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Server not found"
                )
            
            # Build update dict
            update_data = {"updated_at": datetime.now()}
            if updates.name is not None:
                update_data["name"] = updates.name
            if updates.url is not None:
                update_data["url"] = updates.url
            if updates.is_active is not None:
                update_data["is_active"] = updates.is_active
            
            await mcp_servers_collection.update_one(
                {"_id": ObjectId(server_id)},
                {"$set": update_data}
            )
            
            # Return updated server
            updated_server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id)})
            updated_server["_id"] = str(updated_server["_id"])
            updated_server["created_at"] = updated_server["created_at"].isoformat()
            updated_server["updated_at"] = updated_server["updated_at"].isoformat()
            return updated_server
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def delete_server(server_id: str, user_id: str):
        """Delete an MCP server"""
        try:
            # Verify server exists
            server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id)})
            
            if not server:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Server not found"
                )
            
            # Prevent deleting local server
            if server.get("is_local"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot delete local system server"
                )
                
            # Verify ownership
            if server.get("user_id") != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Server not found"
                )
            
            await mcp_servers_collection.delete_one({"_id": ObjectId(server_id)})
            return {"message": "Server deleted"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def test_connection(server_id: str, user_id: str):
        """Test MCP server connection and list available tools"""
        try:
            # Verify server exists and user has access
            server = await mcp_servers_collection.find_one({
                "_id": ObjectId(server_id),
                "$or": [
                    {"user_id": user_id},
                    {"is_local": True}
                ]
            })
            if not server:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Server not found"
                )
            
            # Try to connect to MCP server using connection manager
            try:
                from utils.mcp_connection_manager import mcp_manager
                
                # Use connection manager to get or create connection
                server_url = server["url"]
                
                # Try to connect
                await mcp_manager.connect(server_url)
                
                # Get tools from the connection
                gemini_tools = await mcp_manager.get_tools_for_urls([server_url])
                
                tool_list = []
                if gemini_tools:
                    for tool_group in gemini_tools:
                        if hasattr(tool_group, 'function_declarations') and tool_group.function_declarations:
                            for func in tool_group.function_declarations:
                                tool_list.append({
                                    "name": func.name,
                                    "description": func.description or ""
                                })
                
                if tool_list:
                    return {
                        "status": "connected",
                        "tools": tool_list
                    }
                else:
                    return {
                        "status": "connected",
                        "tools": [],
                        "message": "Connected but no tools available"
                    }
            except Exception as e:
                error_msg = str(e)
                # Provide more helpful error messages
                if "Connection closed" in error_msg or "Connection refused" in error_msg:
                    error_msg = "MCP server is not running or not accessible. This is normal in serverless environments. The server will auto-connect when used in a chat."
                
                return {
                    "status": "error",
                    "error": error_msg,
                    "tools": []
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
