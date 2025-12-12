from fastapi import APIRouter, Depends
from controllers.user_controller import UserController
from core.middleware import get_current_user

router = APIRouter(prefix="/api/users", tags=["Users"])

@router.get("/media")
async def get_user_media(current_user: dict = Depends(get_current_user)):
    """Get all media uploaded by the current user"""
    return await UserController.get_user_media(str(current_user["_id"]))
