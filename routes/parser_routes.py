"""
Parser routes — thin proxy to the PDF-parser microservice (hybrid DocLayout+VLM).

The heavy parsing (torch + DocLayout-YOLO + per-region Gemini) runs in a Docker
microservice on the parser box. This route just forwards the uploaded PDF and
streams the JSON result (markdown + figure images + timing + cost) back to the
frontend Parser page. Kept separate from the RAG ingestion flow on purpose —
this is a standalone testing/eval feature.
"""
import os
import logging

import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from core.middleware import get_current_user
from core.limiter import limiter

router = APIRouter(prefix="/api/parser", tags=["Parser"])
logger = logging.getLogger(__name__)

PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://52.207.56.41/parser")
PARSER_SERVICE_KEY = os.getenv("PARSER_SERVICE_KEY", os.getenv("OMNIROUTE_API_KEY", ""))
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB


@router.get("/health")
async def parser_health(current_user: dict = Depends(get_current_user)):
    """Check whether the parser microservice is reachable, and if region mode is on."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{PARSER_SERVICE_URL}/health",
                headers={"Authorization": f"Bearer {PARSER_SERVICE_KEY}"},
            )
            return {"reachable": r.status_code == 200, "service": r.json() if r.status_code == 200 else None}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


@router.post("/parse")
@limiter.limit("10/minute")
async def parse_pdf(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),        # auto | digital | wholepage | region
    max_pages: int = Form(0),        # 0 = all pages
    judge: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    """Forward a PDF to the parser microservice and return its markdown/cost/timing."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")

    content = await file.read()
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_PDF_BYTES // (1024*1024)} MB")

    try:
        async with httpx.AsyncClient(timeout=600) as client:  # region mode can be slow
            resp = await client.post(
                f"{PARSER_SERVICE_URL}/parse",
                headers={"Authorization": f"Bearer {PARSER_SERVICE_KEY}"},
                files={"file": (file.filename, content, "application/pdf")},
                data={"mode": mode, "max_pages": str(max_pages), "judge": str(judge).lower()},
            )
    except httpx.RequestError as e:
        raise HTTPException(502, f"Parser service unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Parser service error: {resp.text[:300]}")
    return resp.json()


@router.post("/parse/stream")
@limiter.limit("10/minute")
async def parse_pdf_stream(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    max_pages: int = Form(0),
    judge: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    """Proxy the parser service's per-page SSE stream to the frontend."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")
    content = await file.read()
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_PDF_BYTES // (1024*1024)} MB")

    async def relay():
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream(
                    "POST",
                    f"{PARSER_SERVICE_URL}/parse/stream",
                    headers={"Authorization": f"Bearer {PARSER_SERVICE_KEY}"},
                    files={"file": (file.filename, content, "application/pdf")},
                    data={"mode": mode, "max_pages": str(max_pages), "judge": str(judge).lower()},
                ) as resp:
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
        except httpx.RequestError as e:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'error': f'Parser service unreachable: {e}'})}\n\n".encode()

    return StreamingResponse(relay(), media_type="text/event-stream")
