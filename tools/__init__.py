"""
Native Tools Registry
Manages all available native tools
"""
from tools.google_drive.list_folders import ListGoogleDriveFolders
from tools.google_drive.create_folder import CreateGoogleDriveFolder
from tools.utilities.roll_dice import RollDice
from tools.utilities.get_time import GetCurrentTime
from tools.utilities.get_weather import GetWeather
from tools.base import BaseTool
from typing import Dict, List, Optional
from google.genai import types


# Registry of all available tools
AVAILABLE_TOOLS: Dict[str, BaseTool] = {
    # Google Drive tools
    "list_google_drive_folders": ListGoogleDriveFolders(),
    "create_google_drive_folder": CreateGoogleDriveFolder(),
    
    # Utility tools
    "roll_dice": RollDice(),
    "get_current_time": GetCurrentTime(),
    "get_weather": GetWeather(),
}


def get_tool(tool_id: str) -> Optional[BaseTool]:
    """Get a tool by ID"""
    return AVAILABLE_TOOLS.get(tool_id)


def get_all_tools() -> List[BaseTool]:
    """Get all available tools"""
    return list(AVAILABLE_TOOLS.values())


def get_tools_by_category(category: str) -> List[BaseTool]:
    """Get tools by category"""
    return [tool for tool in AVAILABLE_TOOLS.values() if tool.category == category]


def get_gemini_tool_declarations() -> List[types.Tool]:
    """
    Convert all native tools to Gemini Tool format
    Returns a list with a single Tool containing all function declarations
    """
    declarations = [tool.to_gemini_function_declaration() for tool in get_all_tools()]
    return [types.Tool(function_declarations=declarations)]


async def execute_tool(tool_name: str, **kwargs) -> Dict:
    """
    Execute a tool by name
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool parameters
    
    Returns:
        Tool execution result
    """
    tool = get_tool(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    return await tool.execute(**kwargs)
