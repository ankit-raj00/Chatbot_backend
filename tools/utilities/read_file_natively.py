from langchain_core.tools import StructuredTool
from core.database import messages_collection
import logging

logger = logging.getLogger(__name__)

def make_read_file_natively_tool(user_id: str, conversation_id: str):
    
    async def read_file_natively(sandbox_path: str) -> list:
        """Call this tool if you want to load a user's uploaded file natively into your multimodal context.
        Use this for Images, PDFs, and large data files (like CSVs) that you prefer to read directly 
        using your massive context window rather than writing python scripts.
        
        Args:
            sandbox_path: The path of the file, e.g., 'uploads/data.csv' or 'uploads/image.png'.
            
        Returns:
            A multimodal attachment part if successful, or an error string.
        """
        # Find the most recent message in this conversation that has attachments
        cursor = messages_collection.find(
            {"conversation_id": conversation_id, "user_id": user_id, "attachments": {"$exists": True, "$ne": None}}
        ).sort("timestamp", -1).limit(10)
        
        async for msg in cursor:
            attachments = msg.get("attachments", [])
            for att in attachments:
                if att.get("sandbox_path") == sandbox_path:
                    gemini_uri = att.get("gemini_uri")
                    mime_type = att.get("mime_type", "")
                    if gemini_uri:
                        logger.info(f"Loaded {sandbox_path} natively for agent.")
                        return [
                            {"type": "text", "text": f"Successfully loaded {sandbox_path} into context natively."},
                            {"type": "file", "file_id": gemini_uri, "mime_type": mime_type}
                        ]
        
        return f"Error: Could not find a native Gemini File API URI for '{sandbox_path}'. It may have expired or was not uploaded correctly."
        
    return StructuredTool.from_function(
        coroutine=read_file_natively,
        name="read_file_natively",
        description="Load a user's uploaded file natively into your context (for Images, PDFs, etc).",
    )
