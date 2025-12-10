from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class Conversation(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str  # Reference to user
    title: str = "New Conversation"
    mcp_server_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"
    mcp_server_url: Optional[str] = None
