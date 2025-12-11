"""
Google Drive tool: Create folder
"""
from tools.base import BaseTool
from typing import Any, Dict


class CreateGoogleDriveFolder(BaseTool):
    """Create a folder in Google Drive"""
    
    @property
    def name(self) -> str:
        return "create_google_drive_folder"
    
    @property
    def description(self) -> str:
        return "Create a new folder in the user's Google Drive"
    
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
                "folder_name": {
                    "type": "string",
                    "description": "Name of the folder to create"
                }
            },
            "required": ["folder_name"]
        }
    
    async def execute(self, user_id: str, folder_name: str) -> Dict[str, Any]:
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
            
            # Create folder
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            file = service.files().create(
                body=file_metadata,
                fields='id, name, webViewLink'
            ).execute()
            
            return {
                "success": True,
                "folder_id": file.get('id'),
                "folder_name": file.get('name'),
                "link": file.get('webViewLink')
            }
            
        except Exception as e:
            return {"error": f"Failed to create folder: {str(e)}"}
