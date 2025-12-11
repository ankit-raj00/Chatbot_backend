from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse
from controllers.oauth_controller import OAuthController
from controllers.google_oauth_controller import GoogleOAuthController
from core.middleware import get_current_user

router = APIRouter(prefix="/oauth", tags=["OAuth"])

@router.get("/google/authorize")
async def initiate_google_oauth(
    redirect_uri: str = Query(..., description="OAuth redirect URI"),
    current_user: dict = Depends(get_current_user)
):
    """Initiate Google OAuth flow for Drive access"""
    result = await GoogleOAuthController.initiate_oauth(
        user_id=str(current_user["_id"]),
        redirect_uri=redirect_uri
    )
    return RedirectResponse(url=result["oauth_url"])

@router.get("/google/callback")
async def google_oauth_callback(
    code: str = Query(...),
    state: str = Query(...)
):
    """Handle Google OAuth callback for Drive access"""
    import os
    
    # Get backend URL for callback (prod) or default to localhost (dev)
    # Vercel provides VERCEL_URL, but it doesn't include https://
    backend_url = os.getenv("BACKEND_URL")
    if not backend_url:
        vercel_url = os.getenv("VERCEL_URL")
        if vercel_url:
            backend_url = f"https://chatbot-backend-beta-nine.vercel.app"
        else:
            backend_url = "https://chatbot-backend-beta-nine.vercel.app"
            
    redirect_uri = f"{backend_url}/oauth/google/callback"
    
    result = await GoogleOAuthController.handle_callback(code=code, state=state, redirect_uri=redirect_uri)
    
    # Redirect to frontend
    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        return RedirectResponse(url=f"{frontend_url}/mcp-servers?google_auth=success")
    else:
        return {"status": "success", "message": "Google Drive connected successfully (Backend Only Mode)"}

@router.get("/authorize/{server_id}")
async def initiate_oauth(
    server_id: str,
    redirect_uri: str = Query(..., description="OAuth redirect URI"),
    current_user: dict = Depends(get_current_user)
):
    """Initiate OAuth flow for an MCP server"""
    result = await OAuthController.initiate_oauth(
        server_id=server_id,
        user_id=str(current_user["_id"]),
        redirect_uri=redirect_uri
    )
    # Redirect user to OAuth provider
    return RedirectResponse(url=result["oauth_url"])

@router.get("/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...)
):
    """Handle OAuth callback"""
    result = await OAuthController.handle_callback(code=code, state=state)
    # Redirect back to frontend with success
    return RedirectResponse(url=f"/mcp-servers?auth_success=true&server_id={result['server_id']}")

@router.post("/refresh/{server_id}")
async def refresh_token(
    server_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Refresh OAuth token for an MCP server"""
    return await OAuthController.refresh_token(
        server_id=server_id,
        user_id=str(current_user["_id"])
    )

@router.get("/google/status")
async def get_google_oauth_status(
    current_user: dict = Depends(get_current_user)
):
    """Check if user has connected Google Drive"""
    from core.database import oauth_tokens_collection
    
    token_doc = await oauth_tokens_collection.find_one({
        "user_id": str(current_user["_id"]),
        "provider": "google"
    })
    
    if token_doc:
        # Get email from token doc if stored, otherwise return generic response
        return {
            "connected": True,
            "email": token_doc.get("email", "Connected")
        }
    else:
        return {"connected": False}

@router.post("/google/disconnect")
async def disconnect_google_oauth(
    current_user: dict = Depends(get_current_user)
):
    """Disconnect Google Drive by deleting stored credentials"""
    from core.database import oauth_tokens_collection
    
    result = await oauth_tokens_collection.delete_one({
        "user_id": str(current_user["_id"]),
        "provider": "google"
    })
    
    if result.deleted_count > 0:
        return {"success": True, "message": "Google Drive disconnected"}
    else:
        return {"success": False, "message": "No connection found"}
