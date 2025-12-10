from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional, Dict, Any

class MCPServer(BaseModel):
    """MCP Server model for database"""
    id: Optional[str] = Field(None, alias="_id")
    user_id: str  # Reference to user
    name: str
    url: str  # MCP server URL
    is_active: bool = True
    
    # Authentication
    auth_type: str = "none"  # "none" | "oauth" | "api_key"
    oauth_config: Optional[Dict[str, Any]] = None  # OAuth configuration
    access_token: Optional[str] = None  # Encrypted access token
    refresh_token: Optional[str] = None  # Encrypted refresh token
    token_expires_at: Optional[datetime] = None  # Token expiration
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class MCPServerCreate(BaseModel):
    """Request model for creating MCP server"""
    name: str
    url: str
    auth_type: str = "none"
    oauth_config: Optional[Dict[str, Any]] = None

class MCPServerUpdate(BaseModel):
    """Request model for updating MCP server"""
    name: Optional[str] = None
    url: Optional[str] = None
    is_active: Optional[bool] = None
    auth_type: Optional[str] = None
    oauth_config: Optional[Dict[str, Any]] = None
