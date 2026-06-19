"""
Output Routes — serve generated files (PDF, DOCX, PPTX, Excel) for download.

Endpoints:
  GET /api/outputs/list                    — list all generated files for user
  GET /api/outputs/download/{filename}     — download a specific generated file
  DELETE /api/outputs/{filename}           — delete a generated file
"""

import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from core.middleware import get_current_user

router = APIRouter(prefix="/outputs", tags=["Outputs"])

from utils.workspace import workspace_for, WORKSPACE_ROOT as OUTPUTS_DIR

ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html", ".svg", ".png", ".jpg", ".md", ".json"}

def _user_dir(user_id: str) -> Path:
    # Outputs are in the outputs/ subfolder of the user's workspace
    p = workspace_for(user_id) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


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


@router.get("/my/{filename}")
async def download_my_output(
    filename: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Download a file from the current user's workspace.
    User identity comes from JWT — no user_id needed in URL.
    This is the URL that LLM-generated download links should use.
    Example: /api/outputs/my/Dogs_Report.pdf
    """
    user_id   = str(current_user["_id"])
    file_path = _user_dir(user_id) / filename
    
    if not file_path.exists():
        # Fallback to Cloudinary
        from core.database import user_outputs_collection
        from fastapi.responses import RedirectResponse
        output_doc = await user_outputs_collection.find_one({"user_id": user_id, "filename": filename})
        if output_doc and output_doc.get("cloudinary_url"):
            return RedirectResponse(url=output_doc.get("cloudinary_url"))
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
        # Fallback to Cloudinary
        from core.database import user_outputs_collection
        from fastapi.responses import RedirectResponse
        output_doc = await user_outputs_collection.find_one({"user_id": user_id, "filename": filename})
        if output_doc and output_doc.get("cloudinary_url"):
            return RedirectResponse(url=output_doc.get("cloudinary_url"))
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

