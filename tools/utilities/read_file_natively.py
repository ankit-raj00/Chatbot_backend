from langchain_core.tools import StructuredTool
from core.database import messages_collection
import logging

logger = logging.getLogger(__name__)

def make_read_file_natively_tool(user_id: str, conversation_id: str):
    
    async def read_file_natively(sandbox_path: str) -> str:
        """Look up how to access an uploaded file.

        You usually do NOT need this tool:
          - IMAGES the user uploads are shown to you directly in their message —
            just look at them and describe/analyze them. (Images cannot be loaded
            through a tool result on this platform, so this tool will not return
            the picture.)
          - NON-IMAGE files (CSV, PDF, code, data) live in your sandbox — read
            them with sandbox_run_python / sandbox_run_shell.

        This tool only returns guidance text pointing you at the right approach.

        Args:
            sandbox_path: The path of the file, e.g., 'uploads/data.csv'.
        """
        cursor = messages_collection.find(
            {"conversation_id": conversation_id, "user_id": user_id, "attachments": {"$exists": True, "$ne": None}}
        ).sort("timestamp", -1).limit(50)

        async for msg in cursor:
            for att in msg.get("attachments", []):
                att_path = att.get("sandbox_path", "")
                att_name = att.get("original_name", "")
                if att_path == sandbox_path or att_name == sandbox_path or att_path.endswith(f"/{sandbox_path}") or sandbox_path.endswith(f"/{att_name}"):
                    mime_type = (att.get("mime_type") or "")
                    is_image = mime_type.startswith("image/") or att_name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
                    )
                    if is_image:
                        return (
                            f"'{att_name}' is an image. If the user uploaded it in this turn it is "
                            f"already visible to you above — describe/analyze it directly. If you "
                            f"cannot see it (it was uploaded in an earlier turn), ask the user to "
                            f"re-attach it; images cannot be re-loaded through a tool result."
                        )
                    return (
                        f"'{sandbox_path}' is not an image (mime: {mime_type or 'unknown'}). "
                        f"Read it from your sandbox, e.g. "
                        f"sandbox_run_python(\"print(open('{att_path or sandbox_path}').read()[:2000])\") "
                        f"or with pandas for tabular data."
                    )

        return f"Error: Could not find an uploaded file matching '{sandbox_path}'."

    from pydantic import BaseModel, Field
    class ReadFileNativelyInput(BaseModel):
        sandbox_path: str = Field(description="The path of the file, e.g., 'uploads/data.csv'")

    return StructuredTool.from_function(
        coroutine=read_file_natively,
        name="read_file_natively",
        description="Guidance on accessing an uploaded file. Uploaded images are already visible to you directly (no tool needed); non-image files are read via sandbox_run_python/sandbox_run_shell.",
        args_schema=ReadFileNativelyInput
    )
