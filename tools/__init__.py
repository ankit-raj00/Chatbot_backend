"""
Native Tools Registry
Manages all available native tools
"""
from typing import Dict, Optional, Any

# Import Refactored Tools (StructuredTool objects)
from tools.google_drive.list_folders import list_google_drive_folders
from tools.google_drive.create_folder import create_google_drive_folder
from tools.utilities.roll_dice import roll_dice
from tools.utilities.get_time import get_current_time
from tools.utilities.get_weather import get_weather

# Registry of all available tools
# Keys should match the tool name used in the LLM
AVAILABLE_TOOLS: Dict[str, Any] = {
    "list_google_drive_folders": list_google_drive_folders,
    "create_google_drive_folder": create_google_drive_folder,
    "roll_dice": roll_dice,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}

def get_tool(tool_name: str) -> Optional[Any]:
    """Get a tool instance by name"""
    return AVAILABLE_TOOLS.get(tool_name)
    
def get_all_tools() -> list:
    """Get list of all tools"""
    return list(AVAILABLE_TOOLS.values())
