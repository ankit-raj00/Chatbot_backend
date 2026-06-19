"""
ChatController — thin HTTP layer. All business logic lives in ChatService.
This controller handles only file upload processing (Gemini + Cloudinary)
which requires the Gemini client and remains infrastructure-level code.
"""

import os
import json
import logging
import tempfile
from datetime import datetime

from fastapi import HTTPException
from google import genai

from services.chat_service import ChatService
from config.model_config import ModelConfig

import structlog
logger = structlog.get_logger(__name__)


class ChatController:
    """Handles file upload pre-processing, then delegates to ChatService."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=api_key)

    async def process_chat_stream(
        self,
        user_id: str,
        message: str,
        conversation_id: str = None,
        mcp_server_urls: list[str] = None,
        model: str = "gemini-2.5-flash",
        enabled_tools: list[str] = None,
        selected_files: list[str] = None,
        files: list = None,
    ):
        """Thin entry point. Processes file uploads then delegates to ChatService.stream()."""

        # Validate model
        if not ModelConfig.is_valid_model(model):
            model = ModelConfig.DEFAULT_MODEL

        if files and not ModelConfig.supports_images(model):
            yield f"data: {json.dumps({'error': 'Selected model does not support files'})}\n\n"
            return

        # Process file uploads (Gemini Files API + Cloudinary)
        attachments = []
        files_content_parts = []

        if files:
            files_content_parts, attachments = await self._process_uploads(user_id, files)

        # Restore missing outputs from Cloudinary before running the agent
        await self._restore_missing_outputs(user_id)

        # Delegate all streaming logic to ChatService
        async for chunk in ChatService.stream(
            user_id=user_id,
            message=message,
            conversation_id=conversation_id,
            mcp_server_urls=mcp_server_urls,
            model=model,
            enabled_tools=enabled_tools,
            selected_files=selected_files,
            files_content_parts=files_content_parts,
            attachments=attachments,
        ):
            yield chunk

    async def _restore_missing_outputs(self, user_id: str):
        """Restore any missing generated files from Cloudinary before agent runs."""
        from core.database import user_outputs_collection
        from utils.workspace import workspace_for
        from utils.cloudinary_handler import CloudinaryHandler
        
        ws_dir = workspace_for(user_id)
        outputs_dir = ws_dir / "outputs"
        
        try:
            cloudinary = CloudinaryHandler()
            
            # Get all files tracked for this user
            cursor = user_outputs_collection.find({"user_id": user_id})
            async for output_doc in cursor:
                filename = output_doc.get("filename")
                cloudinary_url = output_doc.get("cloudinary_url")
                
                if filename and cloudinary_url:
                    local_path = outputs_dir / filename
                    if not local_path.exists():
                        logger.info(f"Restoring missing file {filename} from Cloudinary")
                        await cloudinary.download_file(cloudinary_url, target_path=str(local_path))
        except Exception as e:
            logger.warning(f"Failed to restore missing outputs from Cloudinary: {e}")

    def _save_to_sandbox(self, user_id: str, filename: str, content: bytes) -> str:
        """Copy uploaded file into the user's sandbox uploads/ dir.
        Returns the path relative to the sandbox root (e.g. 'uploads/report.zip')."""
        from utils.workspace import workspace_for
        
        uploads_dir = workspace_for(user_id) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        dest = uploads_dir / filename
        if dest.exists():
            stem, ext = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = uploads_dir / f"{stem}_{i}{ext}"
                i += 1

        dest.write_bytes(content)
        return f"uploads/{dest.name}"

    async def _process_uploads(self, user_id: str, files: list) -> tuple[list[dict], list[dict]]:
        """Upload files to Gemini Files API and Cloudinary. Returns (content_parts, attachments)."""
        from utils.cloudinary_handler import CloudinaryHandler
        cloudinary = CloudinaryHandler()

        content_parts = []
        attachments = []

        for file_obj in files:
            tmp_path = None
            try:
                suffix = "." + file_obj.filename.split(".")[-1] if "." in file_obj.filename else ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file_obj.read()
                    tmp.write(content)
                    tmp_path = tmp.name

                sandbox_path = self._save_to_sandbox(user_id, file_obj.filename, content)
                cloudinary_url, public_id = await cloudinary.upload_file(tmp_path)
                gemini_file = self.gemini_client.files.upload(file=tmp_path)

                content_parts.append({
                    "type": "file",
                    "file_id": gemini_file.uri,
                    "mime_type": gemini_file.mime_type
                })
                attachments.append({
                    "type": "file",
                    "original_name": file_obj.filename,
                    "mime_type": gemini_file.mime_type,
                    "cloudinary_url": cloudinary_url,
                    "cloudinary_public_id": public_id,
                    "gemini_uri": gemini_file.uri,
                    "gemini_name": gemini_file.name,
                    "gemini_uploaded_at": datetime.now(),
                    "sandbox_path": sandbox_path
                })
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        return content_parts, attachments
