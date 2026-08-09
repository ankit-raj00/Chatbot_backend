"""
ChatController — thin HTTP layer. All business logic lives in ChatService.
This controller handles only file upload pre-processing (Cloudinary + sandbox),
then delegates to ChatService.
"""

import os
import json
import logging
import tempfile
from datetime import datetime

from fastapi import HTTPException

from services.chat_service import ChatService
from config.model_config import ModelConfig
from utils import sandbox_client

import structlog
logger = structlog.get_logger(__name__)


class ChatController:
    """Handles file upload pre-processing, then delegates to ChatService."""

    def __init__(self):
        # No provider client needed here: chat runs via OmniRoute, and uploaded
        # files are stored in Cloudinary + the sandbox (images are read back with
        # their Cloudinary URL via the read_file_natively tool).
        pass

    async def process_chat_stream(
        self,
        user_id: str,
        message: str,
        conversation_id: str = None,
        mcp_server_ids: list[str] = None,
        model: str = None,
        enabled_tools: list[str] = None,
        selected_files: list[str] = None,
        files: list = None,
    ):
        """Thin entry point. Processes file uploads then delegates to ChatService.stream()."""

        # Validate model (defaults to the configured OmniRoute model)
        if not model or not ModelConfig.is_valid_model(model):
            model = ModelConfig.DEFAULT_MODEL

        if files and not ModelConfig.supports_images(model):
            yield f"data: {json.dumps({'error': 'Selected model does not support files'})}\n\n"
            return

        # Resolve the conversation BEFORE touching the sandbox — uploads/outputs
        # are scoped per conversation_id (see utils.workspace.conversation_workspace_for),
        # so we need the real id (not None-meaning-"new") before we can save an
        # uploaded file to the right place. ChatService.stream() would normally
        # create this on a brand-new conversation, but that happens too late for
        # uploads; calling the same idempotent helper here first is harmless —
        # stream()'s own call just becomes an updated_at touch on an existing id.
        conversation_id = await ChatService._ensure_conversation(
            conversation_id, user_id, message, mcp_server_ids
        )

        # Process file uploads (sandbox + Cloudinary)
        attachments = []
        files_content_parts = []

        if files:
            files_content_parts, attachments = await self._process_uploads(user_id, conversation_id, files)

        # Re-hydrate any previously-generated outputs that aren't on local disk
        # (e.g. evicted by cleanup, or gone after a restart). Fire-and-forget +
        # parallel, so it NEVER blocks the first streamed token. The local
        # workspace is a cache; Cloudinary is the durable source of truth.
        from utils.background_tasks import spawn
        from utils.output_store import restore_outputs_background
        spawn(restore_outputs_background(user_id, conversation_id), name="restore_outputs")

        # Delegate all streaming logic to ChatService
        async for chunk in ChatService.stream(
            user_id=user_id,
            message=message,
            conversation_id=conversation_id,
            mcp_server_ids=mcp_server_ids,
            model=model,
            enabled_tools=enabled_tools,
            selected_files=selected_files,
            files_content_parts=files_content_parts,
            attachments=attachments,
        ):
            yield chunk

    async def resume_stream(self, conversation_id: str, user_id: str):
        """Reattach to an in-progress turn for this conversation (e.g. the
        page was reloaded mid-generation) and keep streaming it live."""
        async for chunk in ChatService.resume(conversation_id, user_id):
            yield chunk

    async def check_active_turn(self, conversation_id: str, user_id: str) -> dict:
        """Cheap, non-streaming check for whether this conversation has a
        turn in progress right now — the frontend calls this on page
        load/reload to decide whether to call resume_stream()."""
        return {"active": ChatService.has_active_turn(conversation_id, user_id)}

    async def stop_generation(self, conversation_id: str, user_id: str) -> dict:
        """Cancel this conversation's in-progress turn, if any."""
        stopped = await ChatService.stop_turn(conversation_id, user_id)
        return {"stopped": stopped}

    def _save_to_sandbox(self, user_id: str, conversation_id: str, filename: str, content: bytes) -> str:
        """Copy uploaded file into this conversation's sandbox uploads/ dir.
        Returns the path relative to the sandbox root (e.g. 'uploads/report.zip')."""
        from utils.workspace import conversation_workspace_for

        uploads_dir = conversation_workspace_for(user_id, conversation_id) / "uploads"
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

    async def _process_uploads(self, user_id: str, conversation_id: str, files: list) -> tuple[list[dict], list[dict]]:
        """Save uploads to the sandbox and Cloudinary. Returns (content_parts, attachments).

        content_parts stays empty: the agent SEES images via the analyze_image
        tool (a vision sub-call), not by inlining them into the user message.
        This keeps the mechanism consistent across turns (uploaded images AND
        images the agent generates), works around the gateway's inability to
        take image URLs / tool-message images, and surfaces as an explicit step
        in the UI. Every file is written to the sandbox so analyze_image /
        run_python can reach it by path.
        """
        from utils.cloudinary_handler import CloudinaryHandler
        cloudinary = CloudinaryHandler()

        content_parts = []
        attachments = []

        def _is_image(mime: str | None, name: str) -> bool:
            if mime and mime.startswith("image/"):
                return True
            return name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"))

        for file_obj in files:
            tmp_path = None
            try:
                suffix = "." + file_obj.filename.split(".")[-1] if "." in file_obj.filename else ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file_obj.read()
                    tmp.write(content)
                    tmp_path = tmp.name

                sandbox_path = self._save_to_sandbox(user_id, conversation_id, file_obj.filename, content)

                # Remote mode: code the agent runs executes on the sandbox host,
                # not here, so the upload has to be pushed there too or
                # run_python/analyze_image would see a FileNotFoundError for a
                # file that's sitting right here in the local workspace.
                # Best-effort: a push failure surfaces as a normal missing-file
                # error from the tool call, not a broken upload — the local
                # copy (used by analyze_image, which stays backend-local) is
                # unaffected either way.
                if sandbox_client.is_remote():
                    pushed = await sandbox_client.push_file(user_id, conversation_id, sandbox_path, content)
                    if not pushed:
                        logger.error(f"sandbox push failed for {file_obj.filename} "
                                     f"(conversation {conversation_id}) — remote code execution "
                                     f"won't be able to read this upload")

                cloudinary_url, public_id = None, None
                try:
                    cloudinary_url, public_id = await cloudinary.upload_file(tmp_path)
                except Exception as e:
                    logger.error(f"Cloudinary upload failed for {file_obj.filename}: {e}")

                attachments.append({
                    "type": "file",
                    "original_name": file_obj.filename,
                    "mime_type": file_obj.content_type,
                    "cloudinary_url": cloudinary_url,
                    "cloudinary_public_id": public_id,
                    "sandbox_path": sandbox_path,
                    "is_image": _is_image(file_obj.content_type, file_obj.filename),
                })
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        return content_parts, attachments
