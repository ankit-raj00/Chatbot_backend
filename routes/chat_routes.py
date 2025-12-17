from fastapi import APIRouter, Depends, Form, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from controllers.chat_controller import ChatController
from core.middleware import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize chat controller
chat_controller = ChatController()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    mcp_server_urls: List[str] = []
    model: str = "gemini-2.5-flash"
    enabled_tools: List[str] = []



@router.post("/stream")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Process chat message with streaming response"""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        chat_controller.process_chat_stream(
            user_id=str(current_user["_id"]),
            message=request.message,
            conversation_id=request.conversation_id,
            mcp_server_urls=request.mcp_server_urls,
            model=request.model,
            enabled_tools=request.enabled_tools
        ),
        media_type="text/event-stream"
    )

@router.post("/stream/multimodal")
async def chat_stream_multimodal(
    message: str = Form(...),
    conversation_id: str = Form(None),
    mcp_server_urls: str = Form(None), # JSON string
    model: str = Form("gemini-2.5-flash"),
    images: List[UploadFile] = File(None),
    enabled_tools: str = Form(None), # JSON string
    current_user: dict = Depends(get_current_user)
):
    """Process chat message with images/files + MCP (Streaming)"""
    from fastapi.responses import StreamingResponse
    import json
    
    # Parse JSON fields
    mcp_urls_list = json.loads(mcp_server_urls) if mcp_server_urls else None
    enabled_tools_list = json.loads(enabled_tools) if enabled_tools else None
    
    return StreamingResponse(
        chat_controller.process_chat_stream(
            user_id=str(current_user["_id"]),
            message=message,
            conversation_id=conversation_id,
            mcp_server_urls=mcp_urls_list,
            model=model,
            files=images,
            enabled_tools=enabled_tools_list
        ),
        media_type="text/event-stream"
    )
