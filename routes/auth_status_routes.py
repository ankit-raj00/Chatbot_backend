"""
Auth Status Routes - Check authentication status for tools
"""
from fastapi import APIRouter, Depends
from core.middleware import get_current_user
from core.database import oauth_tokens_collection

router = APIRouter(prefix="/api/auth", tags=["auth-status"])


@router.get("/google-drive/status")
async def check_google_drive_auth(current_user: dict = Depends(get_current_user)):
    """Check if user has authenticated Google Drive"""
    user_id = str(current_user["_id"])
    
    token = await oauth_tokens_collection.find_one({"user_id": user_id})
    
    return {
        "authenticated": token is not None,
        "email": token.get("email") if token else None
    }
