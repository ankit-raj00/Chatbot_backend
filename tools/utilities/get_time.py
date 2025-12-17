"""
Utility tool: Get current time
"""
from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_time():
    """Get the current date and time"""
    now = datetime.now()
    return {
        "result": f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "timestamp": now.isoformat()
    }
