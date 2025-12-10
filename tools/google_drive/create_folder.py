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
        # Import here to avoid circular dependency
        from google_drive_server import execute_create_google_drive_folder
        
        result = await execute_create_google_drive_folder(user_id, folder_name)
        return {"result": result}
