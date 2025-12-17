"""
Setup Node: Initializes/Validates State
"""
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from graph.nodes.common import ChatState

async def setup_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Validation and Setup Node.
    Ensures user_id and context are available.
    """
    # Extract config (injected by controller)
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id")
    enabled_tools = configuration.get("enabled_tools", [])
    
    # Update state with config values (if not already present)
    updates = {}
    if not state.get("user_id") and user_id:
        updates["user_id"] = user_id
        
    if not state.get("enabled_tools"):
        updates["enabled_tools"] = enabled_tools
        
    return updates
