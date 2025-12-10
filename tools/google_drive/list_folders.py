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
        # Import here to avoid circular dependency
        from google_drive_server import execute_list_google_drive_folders
        
        result = await execute_list_google_drive_folders(user_id, page_size)
        return {"result": result}
