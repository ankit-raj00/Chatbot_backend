from fastapi import APIRouter, Depends
from controllers.user_controller import UserController
from core.middleware import get_current_user

router = APIRouter(prefix="/api/users", tags=["Users"])

@router.get("/media")
async def get_user_media(current_user: dict = Depends(get_current_user)):
    """Get all media uploaded by the current user"""
    return await UserController.get_user_media(str(current_user["_id"]))

from services.memory_service import MemoryService

@router.get("/memories")
async def get_user_memories(current_user: dict = Depends(get_current_user)):
    """Get all stored memories for the current user."""
    memories = await MemoryService.get_user_memories(str(current_user["_id"]))
    return {"memories": memories, "count": len(memories)}

@router.delete("/memories")
async def clear_user_memories(current_user: dict = Depends(get_current_user)):
    """Delete all memories for the current user."""
    await MemoryService.clear_user_memories(str(current_user["_id"]))
    return {"message": "All memories cleared"}
