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

router = APIRouter(prefix="/api/outputs", tags=["Outputs"])

_DEFAULT_WS = str(Path.home() / "agentx_workspace")
OUTPUTS_DIR = Path(os.getenv("WORKSPACE_ROOT", _DEFAULT_WS))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html", ".svg", ".png", ".jpg"}

def _user_dir(user_id: str) -> Path:
    p = OUTPUTS_DIR / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/list")
async def list_outputs(current_user: dict = Depends(get_current_user)):
    """List all generated output files for the current user."""
    user_id  = str(current_user["_id"])
    user_dir = _user_dir(user_id)
    files = []
    for f in sorted(user_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
            files.append({
                "filename":   f.name,
                "size_bytes": f.stat().st_size,
                "created_at": f.stat().st_mtime,
                "download_url": f"/api/outputs/download/{user_id}/{f.name}",
            })
    return {"files": files}


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
    if not file_path.exists() or file_path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=404, detail="File not found")

    # Security: no path traversal
    try:
        file_path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@router.delete("/{filename}")
async def delete_output(filename: str, current_user: dict = Depends(get_current_user)):
    """Delete a generated output file."""
    user_id   = str(current_user["_id"])
    file_path = _user_dir(user_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"success": True, "filename": filename}
