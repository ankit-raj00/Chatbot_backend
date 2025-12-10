from fastapi import APIRouter, Depends
from controllers.conversation_controller import ConversationController
from models.conversation import ConversationCreate
from core.middleware import get_current_user

router = APIRouter(prefix="/conversations", tags=["Conversations"])

@router.get("")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    """Get all conversations for the current user"""
    return await ConversationController.get_user_conversations(str(current_user["_id"]))

@router.post("")
async def create_conversation(
    conversation: ConversationCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new conversation"""
    return await ConversationController.create_conversation(
        str(current_user["_id"]),
        conversation
    )

@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all messages for a conversation"""
    return await ConversationController.get_conversation_messages(
        conversation_id,
        str(current_user["_id"])
    )

@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    return await ConversationController.delete_conversation(
        conversation_id,
        str(current_user["_id"])
    )
