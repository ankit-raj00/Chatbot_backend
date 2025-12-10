from fastmcp import FastMCP
from controllers.google_oauth_controller import GoogleOAuthController
from googleapiclient.discovery import build
import sys

mcp = FastMCP(name="Google Drive MCP")

# --- Implementation Logic (Called by ChatController) ---

async def execute_list_google_drive_folders(user_id: str, page_size: int = 100) -> str:
    """
    Actual implementation of listing Google Drive folders.
    """
    try:
        print(f"[DEBUG] execute_list_google_drive_folders called for user: {user_id}")
        
        if not user_id:
            return "Error: User ID is missing."
        
        creds = await GoogleOAuthController.get_user_credentials(user_id)
        if not creds:
            return "Error: Please connect your Google Drive account first."
            
        service = build('drive', 'v3', credentials=creds)
        
        results = service.files().list(
            q="mimeType='application/vnd.google-apps.folder' and trashed=false",
            pageSize=page_size,
            fields="nextPageToken, files(id, name)"
        ).execute()
        
        folders = results.get('files', [])
        
        if not folders:
            return "No folders found in your Google Drive."
            
        folder_list = ", ".join([f"{f['name']}" for f in folders])
        return f"Found {len(folders)} folders: {folder_list}"
        
    except Exception as e:
        print(f"[ERROR] execute_list_google_drive_folders failed: {e}")
        return f"Error listing folders: {str(e)}"

async def execute_create_google_drive_folder(user_id: str, folder_name: str) -> str:
    """
    Actual implementation of creating a Google Drive folder.
    """
    try:
        print(f"[DEBUG] execute_create_google_drive_folder called for user: {user_id}")
        
        if not user_id:
            return "Error: User ID is missing."
            
        if not folder_name:
            return "Error: Folder name is required."
        
        creds = await GoogleOAuthController.get_user_credentials(user_id)
        if not creds:
            return "Error: Please connect your Google Drive account first."
            
        service = build('drive', 'v3', credentials=creds)
        
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        file = service.files().create(body=file_metadata, fields='id').execute()
        return f"✅ Folder '{folder_name}' created successfully with ID: {file.get('id')}"
        
    except Exception as e:
        print(f"[ERROR] execute_create_google_drive_folder failed: {e}")
        return f"Error creating folder: {str(e)}"


# --- MCP Tool Definitions (The Interface for Gemini) ---

@mcp.tool()
async def list_google_drive_folders(page_size: int = 100) -> str:
    """
    List folders in Google Drive.
    Args:
        page_size: Number of folders to return.
    """
    # This is just a placeholder for the tool signature.
    # The actual execution is intercepted by ChatController and routed to execute_list_google_drive_folders above.
    return "This is a placeholder. The actual implementation is handled by the backend."

@mcp.tool()
async def create_google_drive_folder(folder_name: str) -> str:
    """
    Create a new folder in Google Drive.
    Args:
        folder_name: Name of the new folder.
    """
    # This is just a placeholder for the tool signature.
    # The actual execution is intercepted by ChatController and routed to execute_create_google_drive_folder above.
    return "This is a placeholder. The actual implementation is handled by the backend."

if __name__ == "__main__":
    mcp.run()
