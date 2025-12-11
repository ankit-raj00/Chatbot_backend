"""
Google Drive tool: List folders
"""
from tools.base import BaseTool
from typing import Any, Dict


class ListGoogleDriveFolders(BaseTool):
    """List folders in Google Drive"""
    
    @property
    def name(self) -> str:
        return "list_google_drive_folders"
    
    @property
    def description(self) -> str:
        return "List folders in the user's Google Drive"
    
    @property
    def category(self) -> str:
        return "google_drive"
    
    @property
    def requires_auth(self) -> bool:
        return True
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_size": {
                    "type": "integer",
                    "description": "Number of folders to return (default: 100)",
                    "default": 100
                }
            }
        }
    
    async def execute(self, user_id: str, page_size: int = 100) -> Dict[str, Any]:
        """Execute the tool"""
        from controllers.google_oauth_controller import GoogleOAuthController
        from googleapiclient.discovery import build
        
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
