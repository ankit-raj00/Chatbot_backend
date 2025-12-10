from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

class Message(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    conversation_id: str
    user_id: str  # Reference to user
    role: str  # 'user' or 'assistant'
    content: str
    attachments: Optional[List[Dict[str, Any]]] = None  # Image/file attachments
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    attachments: Optional[List[Dict[str, Any]]] = None
