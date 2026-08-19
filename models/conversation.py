from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal

class Conversation(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str  # Reference to user
    title: str = "New Conversation"
    mcp_server_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"
    mcp_server_id: Optional[str] = None


class MessageFeedback(BaseModel):
    """Thumbs up/down on one assistant message. `rating=None` clears it,
    which is how the UI toggles a rating back off. `reason` is the optional
    free-text captured on a thumbs-down — it's what makes the exported
    dataset triageable rather than just a count."""
    rating: Optional[Literal["up", "down"]] = None
    reason: Optional[str] = None


class ConversationRename(BaseModel):
    """Rename a conversation. Trimmed + length-capped server-side so a stray
    paste can't write an unbounded title into the sidebar."""
    title: str = Field(..., min_length=1, max_length=200)
