"""
Google Drive tool: List folders
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class ListFoldersArgs(BaseModel):
    page_size: int = Field(default=100, description="Number of folders to return (default: 100)")

@tool(args_schema=ListFoldersArgs)
async def list_google_drive_folders(page_size: int = 100, user_id: Optional[str] = None) -> Dict[str, Any]:
    """List folders in the user's Google Drive"""
    from controllers.google_oauth_controller import GoogleOAuthController
    from googleapiclient.discovery import build
    
    if not user_id:
        return {"error": "Authentication required: user_id missing from tool context"}
    
    # Get user credentials
    creds = await GoogleOAuthController.get_user_credentials(user_id)
    if not creds:
        return {"error": "User not authenticated for Google Drive. Please connect your account in the tools menu."}
        
    try:
        # Build Drive service
        service = build('drive', 'v3', credentials=creds)
        
        # List folders
        results = service.files().list(
            pageSize=page_size,
            q="mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields="nextPageToken, files(id, name, createdTime, webViewLink)",
            orderBy="createdTime desc"
        ).execute()
        
        folders = results.get('files', [])
        return {"folders": folders, "count": len(folders)}
        
    except Exception as e:
        return {"error": f"Failed to list folders: {str(e)}"}
