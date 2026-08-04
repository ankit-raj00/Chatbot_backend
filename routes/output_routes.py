"""
Output Routes — serve generated files (PDF, DOCX, PPTX, Excel) for download.

Endpoints:
  GET /api/outputs/list                    — list all generated files for user
  GET /api/outputs/download/{filename}     — download a specific generated file
  DELETE /api/outputs/{filename}           — delete a generated file
"""

import os
import mimetypes
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from core.middleware import get_current_user

router = APIRouter(prefix="/outputs", tags=["Outputs"])

from utils.workspace import workspace_for, conversation_workspace_for, WORKSPACE_ROOT as OUTPUTS_DIR

ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html", ".svg", ".png", ".jpg", ".md", ".json"}

def _user_dir(user_id: str) -> Path:
    # LEGACY: outputs/ used to be flat, shared across all of a user's
    # conversations. Kept around only so files created before the
    # per-conversation migration (see conversation_workspace_for) still
    # resolve. New files are written to _conversation_dir instead.
    p = workspace_for(user_id) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _conversation_dir(user_id: str, conversation_id: str) -> Path:
    p = conversation_workspace_for(user_id, conversation_id) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_legacy_by_filename(user_id: str, filename: str) -> Path | None:
    """Best-effort fallback for callers that only have a filename, not a
    conversation_id (old markdown links the agent wrote inline, or old
    saved messages from before conversation-scoped URLs existed). Checks the
    legacy flat dir first, then the most-recently-modified match across this
    user's conversation folders. Ambiguous by construction if two
    conversations independently created a same-named file — prefer the
    conversation-scoped route (/outputs/my/{conversation_id}/{filename})
    whenever a conversation_id is available."""
    legacy = _user_dir(user_id) / filename
    if legacy.exists():
        return legacy

    conv_root = workspace_for(user_id) / "conversations"
    if not conv_root.exists():
        return None
    candidates = [
        p for p in conv_root.glob(f"*/outputs/{filename}") if p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


async def _serve_cloudinary(cloudinary_url: str, filename: str) -> StreamingResponse:
    """
    Stream a file's bytes from Cloudinary THROUGH this backend (same-origin).

    WHY: a 302 redirect to Cloudinary breaks in-browser preview — a cross-origin
    `fetch(..., {credentials:'include'})` can't read Cloudinary's response (no CORS
    headers), even though a plain <a> download follows the redirect fine. Proxying
    the bytes keeps preview AND download working, and lazily re-hydrates files that
    were evicted from the local cache (e.g. after a server restart/redeploy).
    """
    import httpx

    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    async def _iter():
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async with client.stream("GET", cloudinary_url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    yield chunk

    return StreamingResponse(
        _iter(),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.get("/list")
async def list_outputs(current_user: dict = Depends(get_current_user)):
    """List all generated output files for the current user."""
    user_id  = str(current_user["_id"])
    user_dir = _user_dir(user_id)
    files_dict = {}
    
    from core.database import user_outputs_collection
    cursor = user_outputs_collection.find({"user_id": user_id})
    async for output_doc in cursor:
        filename = output_doc.get("filename")
        if filename:
            dt = output_doc.get("updated_at") or output_doc.get("created_at")
            files_dict[filename] = {
                "filename": filename,
                "size_bytes": output_doc.get("size_bytes", 0),
                "download_url": f"/outputs/download/{user_id}/{filename}",
                "created_at": dt.timestamp() if hasattr(dt, "timestamp") else 0
            }
            
    if user_dir.exists():
        for f in user_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                files_dict[f.name] = {
                    "filename":   f.name,
                    "size_bytes": f.stat().st_size,
                    "created_at": f.stat().st_mtime,
                    "download_url": f"/outputs/download/{user_id}/{f.name}",
                }
                
    files = list(files_dict.values())
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"files": files}

@router.get("/my")
async def list_my_outputs(current_user: dict = Depends(get_current_user)):
    """List all generated files for the current user (JWT auth only, no user_id in URL)."""
    user_id  = str(current_user["_id"])
    user_dir = _user_dir(user_id)
    files_dict = {}
    
    # 1. Get from MongoDB (Cloudinary files)
    from core.database import user_outputs_collection
    cursor = user_outputs_collection.find({"user_id": user_id})
    async for output_doc in cursor:
        filename = output_doc.get("filename")
        if filename:
                dt = output_doc.get("updated_at") or output_doc.get("created_at")
                files_dict[filename] = {
                    "filename": filename,
                    "size_bytes": output_doc.get("size_bytes", 0),
                    "download_url": f"/outputs/my/{filename}",
                    "created_at": dt.timestamp() if hasattr(dt, "timestamp") else 0
                }
            
    # 2. Get from local disk (may have newer/un-uploaded files)
    if user_dir.exists():
        for f in user_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                files_dict[f.name] = {
                    "filename":     f.name,
                    "size_bytes":   f.stat().st_size,
                    "download_url": f"/outputs/my/{f.name}",
                    "created_at":   f.stat().st_mtime,
                }
                
    files = list(files_dict.values())
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"files": files}


@router.get("/my/{conversation_id}/{filename}")
async def download_my_conversation_output(
    conversation_id: str,
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Download a file generated in a specific conversation. This is the
    unambiguous, correct URL — what newly-generated files_created entries
    use — since two different conversations can independently create a
    same-named file (e.g. two unrelated "report.pdf"), and only the
    conversation_id disambiguates which one you mean.
    """
    user_id   = str(current_user["_id"])
    file_path = _conversation_dir(user_id, conversation_id) / filename

    if not file_path.exists():
        from core.database import user_outputs_collection
        output_doc = await user_outputs_collection.find_one(
            {"user_id": user_id, "conversation_id": conversation_id, "filename": filename}
        )
        if output_doc and output_doc.get("cloudinary_url"):
            return await _serve_cloudinary(output_doc["cloudinary_url"], filename)
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    if file_path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="File type not permitted")
    try:
        file_path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        content_disposition_type="inline"
    )


@router.get("/my/{filename}")
async def download_my_output(
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """
    LEGACY fallback: download a file by name alone, with no conversation_id.
    Used only by (a) files created before the per-conversation migration and
    (b) UI code paths that don't have a conversation_id handy (inline
    markdown links the agent wrote, attachment-preview fallbacks). Ambiguous
    if two conversations independently created a same-named file — picks the
    most recently modified match. Prefer
    /outputs/my/{conversation_id}/{filename} wherever a conversation_id is
    available (that's what newly-generated files_created entries use).
    """
    user_id   = str(current_user["_id"])
    file_path = _find_legacy_by_filename(user_id, filename)

    if not file_path:
        # Local cache miss — stream from Cloudinary through the backend (same-origin
        # so browser preview works, not a cross-origin 302 that CORS blocks).
        # No conversation_id to disambiguate — take the most recently updated match.
        from core.database import user_outputs_collection
        output_doc = await user_outputs_collection.find_one(
            {"user_id": user_id, "filename": filename}, sort=[("updated_at", -1)]
        )
        if output_doc and output_doc.get("cloudinary_url"):
            return await _serve_cloudinary(output_doc["cloudinary_url"], filename)
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    if file_path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="File type not permitted")
    try:
        file_path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        content_disposition_type="inline"
    )


@router.get("/download/{user_id}/{filename}")
async def download_output(
    user_id: str,
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """Download a generated output file. Users can only download their own files."""
    requester_id = str(current_user["_id"])
    # Admin can download any file; users can only download their own
    if requester_id != user_id and not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = _user_dir(user_id) / filename
    if not file_path.exists():
        # Local cache miss — stream from Cloudinary through the backend (same-origin).
        from core.database import user_outputs_collection
        output_doc = await user_outputs_collection.find_one({"user_id": user_id, "filename": filename})
        if output_doc and output_doc.get("cloudinary_url"):
            return await _serve_cloudinary(output_doc["cloudinary_url"], filename)
        raise HTTPException(status_code=404, detail="File not found")

    if file_path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=404, detail="File not found")

    # Security: no path traversal
    try:
        file_path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        content_disposition_type="inline"
    )


@router.delete("/sandbox")
async def reset_sandbox(current_user: dict = Depends(get_current_user)):
    """Wipe the current user's entire sandbox (uploads, outputs, work, venv, caches).
    Use if the environment gets into a bad state (corrupted venv, disk full, etc.)."""
    import shutil
    user_id = str(current_user["_id"])
    ws = workspace_for(user_id)
    if ws.exists():
        shutil.rmtree(ws)
    return {"success": True, "message": "Sandbox reset. A fresh environment will be created on next use."}


@router.delete("/{filename}")
async def delete_output(filename: str, current_user: dict = Depends(get_current_user)):
    """Delete a generated output file."""
    user_id   = str(current_user["_id"])
    file_path = _user_dir(user_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"success": True, "filename": filename}

