"""
Utility tool: Get current time
"""
from tools.base import BaseTool
from typing import Any, Dict
from datetime import datetime


class GetCurrentTime(BaseTool):
    """Get the current date and time"""
    
    @property
    def name(self) -> str:
        return "get_current_time"
    
    @property
    def description(self) -> str:
        return "Get the current date and time"
    
    @property
    def category(self) -> str:
        return "utilities"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {}
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the tool"""
        now = datetime.now()
        return {
            "result": f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            "timestamp": now.isoformat()
        }
