"""
Google Drive tool: Create folder
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class CreateFolderArgs(BaseModel):
    folder_name: str = Field(description="Name of the folder to create")
    user_id: Optional[str] = Field(None, description="User ID (injected automatically)")

@tool(args_schema=CreateFolderArgs)
async def create_google_drive_folder(folder_name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a new folder in the user's Google Drive"""
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
