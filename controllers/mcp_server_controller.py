"""
Controller for MCP server CRUD + connection testing.

All structured connection fields (transport, command/args/env, url, headers,
auth_type) are actually persisted and used to build real connections via
utils/mcp_connection_manager.py — unlike the previous version, which accepted
these fields in the request models but silently dropped them before writing to
MongoDB.
"""
from fastapi import HTTPException, status
from core.database import mcp_servers_collection
from models.mcp_server import MCPServerCreate, MCPServerUpdate
from datetime import datetime
from bson import ObjectId

import structlog
logger = structlog.get_logger(__name__)


def _serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    for field in ("created_at", "updated_at"):
        if isinstance(doc.get(field), datetime):
            doc[field] = doc[field].isoformat()
        elif field not in doc:
            doc[field] = doc.get("created_at") or datetime.now().isoformat()
    # Never send secret values to the client — presence flags only.
    if doc.get("headers"):
        doc["headers"] = {k: "••••••••" for k in doc["headers"]}
    if doc.get("env"):
        doc["env"] = {k: "••••••••" for k in doc["env"]}
    return doc


class MCPServerController:
    """Controller for MCP server operations"""

    @staticmethod
    async def get_user_servers(user_id: str):
        """Get all MCP servers for a user (+ shared local servers)."""
        try:
            servers_cursor = mcp_servers_collection.find({
                "$or": [{"user_id": user_id}, {"is_local": True}]
            }).sort("created_at", -1)
            servers_list = await servers_cursor.to_list(length=100)
            return [_serialize(s) for s in servers_list]
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @staticmethod
    async def add_server(user_id: str, server: MCPServerCreate):
        """Add a new MCP server with its full structured connection config."""
        from utils.crypto import encrypt_dict_values

        try:
            new_server = {
                "user_id": user_id,
                "name": server.name,
                "is_active": True,
                "is_local": False,
                "transport": server.transport,
                "command": server.command,
                "args": server.args,
                "env": encrypt_dict_values(server.env),
                "url": server.url,
                "headers": encrypt_dict_values(server.headers) if server.auth_type == "headers" else None,
                "auth_type": server.auth_type,
                "oauth_authorized": False,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            result = await mcp_servers_collection.insert_one(new_server)
            new_server["_id"] = result.inserted_id
            return _serialize(new_server)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @staticmethod
    async def update_server(server_id: str, user_id: str, updates: MCPServerUpdate):
        """Update an MCP server. Reconnects lazily on next use if config changed
        (utils.mcp_connection_manager fingerprints the config automatically)."""
        from utils.crypto import encrypt_dict_values
        from utils.mcp_connection_manager import mcp_manager

        try:
            server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id)})
            if not server:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
            if server.get("is_local"):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot update local system server")
            if server.get("user_id") != user_id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

            update_data = {"updated_at": datetime.now()}
            for field in ("name", "is_active", "transport", "command", "args", "url", "auth_type"):
                val = getattr(updates, field, None)
                if val is not None:
                    update_data[field] = val
            if updates.env is not None:
                update_data["env"] = encrypt_dict_values(updates.env)
            if updates.headers is not None:
                update_data["headers"] = encrypt_dict_values(updates.headers)
            # Changing auth_type away from oauth invalidates any prior authorization.
            if updates.auth_type is not None and updates.auth_type != server.get("auth_type"):
                update_data["oauth_authorized"] = False

            await mcp_servers_collection.update_one({"_id": ObjectId(server_id)}, {"$set": update_data})
            mcp_manager.invalidate_tool_cache(user_id, server_id)

            updated_server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id)})
            return _serialize(updated_server)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @staticmethod
    async def delete_server(server_id: str, user_id: str):
        """Delete an MCP server and disconnect its live connection immediately —
        otherwise a stale client/cached tools would keep serving until restart."""
        from utils.mcp_connection_manager import mcp_manager
        from core.database import mcp_oauth_tokens_collection

        try:
            server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id)})
            if not server:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
            if server.get("is_local"):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete local system server")
            if server.get("user_id") != user_id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

            await mcp_servers_collection.delete_one({"_id": ObjectId(server_id)})
            await mcp_oauth_tokens_collection.delete_one({"user_id": user_id, "server_id": server_id})
            await mcp_manager.disconnect(user_id, server_id)
            return {"message": "Server deleted"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @staticmethod
    async def test_connection(server_id: str, user_id: str):
        """Test MCP server connection and list available tools. Real error
        messages are always surfaced — never rewritten into a guess."""
        from utils.mcp_connection_manager import mcp_manager

        server = await mcp_servers_collection.find_one({
            "_id": ObjectId(server_id),
            "$or": [{"user_id": user_id}, {"is_local": True}],
        })
        if not server:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")

        return await mcp_manager.test_connection(user_id, server)

    @staticmethod
    async def initiate_oauth(server_id: str, user_id: str, redirect_uri: str) -> str:
        """Start the OAuth 2.1 authorization flow (discovery + dynamic client
        registration + PKCE, all handled by the MCP SDK). Returns the auth URL
        the user's browser should be sent to."""
        from utils.mcp_oauth_flow import initiate_authorize

        server = await mcp_servers_collection.find_one({"_id": ObjectId(server_id), "user_id": user_id})
        if not server:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
        if server.get("auth_type") != "oauth":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Server is not configured for OAuth")
        if not server.get("url"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Server has no URL configured")

        try:
            result = await initiate_authorize(user_id, server_id, server["url"], redirect_uri)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

        if not result.get("oauth_url"):
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Server did not provide an authorization URL")
        return result["oauth_url"]

    @staticmethod
    async def oauth_status(server_id: str, user_id: str) -> dict:
        """Poll target for the frontend while a user completes the third-party
        login in another tab."""
        server = await mcp_servers_collection.find_one(
            {"_id": ObjectId(server_id), "user_id": user_id},
            {"oauth_authorized": 1, "oauth_last_error": 1},
        )
        if not server:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
        return {
            "authorized": bool(server.get("oauth_authorized")),
            "error": server.get("oauth_last_error"),
        }
