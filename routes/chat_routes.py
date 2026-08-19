from fastapi import APIRouter, Depends, Form, File, UploadFile, Request
from pydantic import BaseModel
from typing import List, Optional
from controllers.chat_controller import ChatController
from core.middleware import get_current_user
from core.limiter import limiter
from config.model_config import ModelConfig
import os

router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize chat controller
chat_controller = ChatController()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    mcp_server_ids: List[str] = []
    model: str = ModelConfig.DEFAULT_MODEL
    enabled_tools: List[str] = []
    selected_files: List[str] = []



@router.post("/stream")
@limiter.limit(f"{os.getenv('CHAT_RATE_LIMIT', '20')}/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process chat message with streaming response"""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        chat_controller.process_chat_stream(
            user_id=str(current_user["_id"]),
            message=chat_request.message,
            conversation_id=chat_request.conversation_id,
            mcp_server_ids=chat_request.mcp_server_ids,
            model=chat_request.model,
            enabled_tools=chat_request.enabled_tools,
            selected_files=chat_request.selected_files
        ),
        media_type="text/event-stream"
    )

@router.post("/stream/multimodal")
@limiter.limit(f"{os.getenv('CHAT_RATE_LIMIT', '20')}/minute")
async def chat_stream_multimodal(
    request: Request,
    message: str = Form(...),
    conversation_id: str = Form(None),
    mcp_server_ids: str = Form(None), # JSON string
    model: str = Form(ModelConfig.DEFAULT_MODEL),
    images: List[UploadFile] = File(None),
    enabled_tools: str = Form(None), # JSON string
    selected_files: str = Form(None), # JSON string
    current_user: dict = Depends(get_current_user)
):
    """Process chat message with images/files + MCP (Streaming)"""
    from fastapi.responses import StreamingResponse
    import json

    # Parse JSON fields
    mcp_ids_list = json.loads(mcp_server_ids) if mcp_server_ids else None
    enabled_tools_list = json.loads(enabled_tools) if enabled_tools else None
    selected_files_list = json.loads(selected_files) if selected_files else None

    return StreamingResponse(
        chat_controller.process_chat_stream(
            user_id=str(current_user["_id"]),
            message=message,
            conversation_id=conversation_id,
            mcp_server_ids=mcp_ids_list,
            model=model,
            files=images,
            enabled_tools=enabled_tools_list,
            selected_files=selected_files_list
        ),
        media_type="text/event-stream"
    )


class RetryRequest(BaseModel):
    # All optional: an unspecified model falls back to whichever produced the
    # response being replaced (see ChatService.retry), the rest to the current
    # composer selection.
    mcp_server_ids: List[str] = []
    model: str | None = None
    enabled_tools: List[str] = []
    selected_files: List[str] = []


@router.post("/{conversation_id}/messages/{message_id}/retry")
@limiter.limit(f"{os.getenv('CHAT_RATE_LIMIT', '20')}/minute")
async def retry_message(
    request: Request,
    conversation_id: str,
    message_id: str,
    retry_request: RetryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Regenerate the conversation's most recent assistant response, replacing
    it in place. Same SSE wire format as /stream."""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        chat_controller.retry_stream(
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=str(current_user["_id"]),
            mcp_server_ids=retry_request.mcp_server_ids,
            model=retry_request.model,
            enabled_tools=retry_request.enabled_tools,
            selected_files=retry_request.selected_files,
        ),
        media_type="text/event-stream"
    )


@router.get("/{conversation_id}/active-turn")
@limiter.limit("60/minute")
async def get_active_turn(
    request: Request,
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Lightweight check: is a generation currently in progress for this
    conversation? Call this on page load/reload before resume_stream so a
    reload with nothing in progress doesn't open a pointless SSE connection."""
    return await chat_controller.check_active_turn(conversation_id, str(current_user["_id"]))


@router.get("/{conversation_id}/resume")
@limiter.limit("30/minute")
async def resume_chat_stream(
    request: Request,
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Reattach to an in-progress turn (e.g. after a page reload mid-generation)
    and keep streaming it — replays everything already generated, then
    continues live, same wire format as /stream."""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        chat_controller.resume_stream(conversation_id, str(current_user["_id"])),
        media_type="text/event-stream"
    )


@router.post("/{conversation_id}/stop")
@limiter.limit("30/minute")
async def stop_chat_generation(
    request: Request,
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Cancel this conversation's in-progress turn, if any."""
    return await chat_controller.stop_generation(conversation_id, str(current_user["_id"]))
