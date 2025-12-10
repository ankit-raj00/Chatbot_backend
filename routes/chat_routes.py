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

@router.post("")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Process chat message with Gemini + MCP"""
    return await chat_controller.process_chat(
        user_id=str(current_user["_id"]),
        message=request.message,
        conversation_id=request.conversation_id,
        mcp_server_url=request.mcp_server_url
    )

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
            model=request.model
        ),
        media_type="text/event-stream"
    )

@router.post("/stream/multimodal")
async def chat_stream_multimodal(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    mcp_server_urls: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash"),
    images: List[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    """Process multimodal chat message with streaming response"""
    from fastapi.responses import StreamingResponse
    import json
    
    # Parse mcp_server_urls from JSON string
    parsed_urls = []
    if mcp_server_urls:
        try:
            parsed_urls = json.loads(mcp_server_urls)
        except:
            parsed_urls = []
    
    return StreamingResponse(
        chat_controller.process_chat_stream_multimodal(
            user_id=str(current_user["_id"]),
            message=message,
            conversation_id=conversation_id,
            mcp_server_urls=parsed_urls,
            model=model,
            images=images
        ),
        media_type="text/event-stream"
    )
