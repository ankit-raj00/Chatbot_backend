"""
Tool Controller - Manage native tools
"""
from tools import get_all_tools, get_tool, get_tools_by_category
from typing import List, Dict, Any


class ToolController:
    """Controller for native tools management"""
    
    @staticmethod
    async def get_all_tools() -> Dict[str, Any]:
        """
        Get all available native tools
        
        Returns:
            Dict with tools list
        """
        tools = get_all_tools()
        
        return {
            "tools": [
                {
                    "tool_id": tool.name,
                    "name": tool.name.replace("_", " ").title(),
                    "description": tool.description,
                    "category": tool.category,
                    "requires_auth": tool.requires_auth,
                    "is_enabled": True
                }
                for tool in tools
            ]
        }
    
    @staticmethod
    async def get_tools_by_category(category: str) -> Dict[str, Any]:
        """
        Get tools by category
        
        Args:
            category: Tool category (e.g., 'google_drive', 'utilities')
        
        Returns:
            Dict with tools list
        """
        tools = get_tools_by_category(category)
        
        return {
            "category": category,
            "tools": [
                {
                    "tool_id": tool.name,
                    "name": tool.name.replace("_", " ").title(),
                    "description": tool.description,
                    "requires_auth": tool.requires_auth
                }
                for tool in tools
            ]
        }
