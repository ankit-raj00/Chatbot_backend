"""
Native Tools Registry
Manages all available native tools
"""
from typing import Dict, Optional, Any, List

# Import Tools
from tools.google_drive.list_folders import list_google_drive_folders
from tools.google_drive.create_folder import create_google_drive_folder
from tools.utilities.roll_dice import roll_dice
from tools.utilities.get_time import get_current_time
from tools.utilities.get_weather import get_weather

# Registry
AVAILABLE_TOOLS: Dict[str, Any] = {
    "list_google_drive_folders": list_google_drive_folders,
    "create_google_drive_folder": create_google_drive_folder,
    "roll_dice": roll_dice,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}

# Metadata Registry to support legacy Controller fields
TOOL_METADATA = {
    "list_google_drive_folders": {"category": "google_drive", "requires_auth": True},
    "create_google_drive_folder": {"category": "google_drive", "requires_auth": True},
    "roll_dice": {"category": "utilities", "requires_auth": False},
    "get_current_time": {"category": "utilities", "requires_auth": False},
    "get_weather": {"category": "utilities", "requires_auth": False},
}

def get_tool(tool_name: str) -> Optional[Any]:
    """Get a tool instance by name"""
    return AVAILABLE_TOOLS.get(tool_name)
    
def get_all_tools() -> list:
    """Get list of all tools with injected metadata for Controller compatibility"""
    tools = []
    for name, tool in AVAILABLE_TOOLS.items():
        # Monkey-patch metadata for the controller's benefit
        # (The controller expects .category and .requires_auth)
        metadata = TOOL_METADATA.get(name, {})
        tool.category = metadata.get("category", "general")
        tool.requires_auth = metadata.get("requires_auth", False)
        tools.append(tool)
    return tools

def get_tools_by_category(category: str) -> List[Any]:
    """Get tools by category"""
    filtered_tools = []
    for name, tool in AVAILABLE_TOOLS.items():
        metadata = TOOL_METADATA.get(name, {})
        if metadata.get("category") == category:
            tool.category = category
            tool.requires_auth = metadata.get("requires_auth", False)
            filtered_tools.append(tool)
    return filtered_tools
