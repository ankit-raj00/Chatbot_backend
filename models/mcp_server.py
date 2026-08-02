"""
MCP Server connection configuration.

`transport` is explicit (never guessed from a URL string) so any real-world MCP
server can be connected precisely:
  - "stdio": local subprocess — command, args, env
  - "sse":   legacy HTTP+SSE transport (protocol 2024-11-05, deprecated but still
             run by some older servers)
  - "http":  Streamable HTTP (current spec, protocol 2025-06-18) — url, headers

`auth_type`:
  - "none":    no authentication
  - "headers": arbitrary static HTTP headers sent on every request (covers the
               common "Authorization: Bearer <token>" / API-key-header case).
               Header VALUES are encrypted at rest.
  - "oauth":   OAuth 2.1 (PKCE + RFC 9728/8414 discovery + RFC 7591 dynamic
               client registration). No manual client_id/secret entry — those
               are auto-discovered/registered. See utils/mcp_oauth_flow.py.
               Authorization state (tokens, client_info) lives in the separate
               mcp_oauth_tokens collection, not on this document.
"""
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Optional, Dict, List, Literal

Transport = Literal["stdio", "sse", "http"]
AuthType = Literal["none", "headers", "oauth"]


class MCPServer(BaseModel):
    """MCP Server model for database"""
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    name: str
    is_active: bool = True
    is_local: bool = False   # shared system server — not user-owned, not editable/deletable via API

    transport: Transport = "http"

    # stdio
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None   # values encrypted at rest

    # sse / http
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None   # values encrypted at rest when auth_type == "headers"

    auth_type: AuthType = "none"
    oauth_authorized: bool = False   # set True once a token exists in mcp_oauth_tokens_collection

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class MCPServerCreate(BaseModel):
    """Request model for creating an MCP server."""
    name: str
    transport: Transport = "http"

    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None

    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    auth_type: AuthType = "none"

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerCreate":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("'command' is required for stdio transport")
        else:
            if not self.url:
                raise ValueError("'url' is required for sse/http transport")
        if self.auth_type == "headers" and not self.headers:
            raise ValueError("'headers' must be provided when auth_type is 'headers'")
        return self


class MCPServerUpdate(BaseModel):
    """Request model for updating an MCP server. Only provided fields change."""
    name: Optional[str] = None
    is_active: Optional[bool] = None

    transport: Optional[Transport] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    auth_type: Optional[AuthType] = None
