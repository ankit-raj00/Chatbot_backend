"""
Client for the hybrid PDF parser microservice (DocLayout-YOLO + VLM, on the
parser box). The service is stateless: PDF in → typed blocks out. All
chunking, embedding and indexing stays here in the backend, so the parser
never needs database credentials or tenancy logic.
"""
import os
import time
import logging
from typing import Any, Dict, Optional

import httpx
import structlog

from services.parser_dashboard_service import ParserDashboardService

logger = structlog.get_logger(__name__)

PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://52.207.56.41/parser")
PARSER_SERVICE_KEY = os.getenv("PARSER_SERVICE_KEY", os.getenv("OMNIROUTE_API_KEY", ""))
PARSER_TIMEOUT = float(os.getenv("PARSER_TIMEOUT", "1800"))   # big scans are slow


class ParserServiceError(RuntimeError):
    pass


async def parse_pdf_blocks(
    file_path: str,
    mode: str = "auto",
    max_pages: int = 0,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Send a PDF to the parser service and return its structured result:
        {"markdown": str, "blocks": [...], "images": [...], "pages": int, ...}

    `blocks` is the typed structure consumed by rag.chunking.block_chunker.
    Raises ParserServiceError on failure so the caller can fall back.
    """
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as fh:
        content = fh.read()

    t0 = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout or PARSER_TIMEOUT) as client:
            resp = await client.post(
                f"{PARSER_SERVICE_URL}/parse",
                headers={"Authorization": f"Bearer {PARSER_SERVICE_KEY}"},
                files={"file": (filename, content, "application/pdf")},
                # ingestion only needs the Cloudinary URL, not fat base64 crops
                data={"mode": mode, "max_pages": str(max_pages), "include_b64": "false"},
            )
    except httpx.RequestError as e:
        await ParserDashboardService.record_run(
            filename=filename, source="rag_ingest", status="failed", mode=mode,
            bytes_size=len(content), duration_ms=int((time.perf_counter() - t0) * 1000),
            error=f"parser service unreachable: {e}",
        )
        raise ParserServiceError(f"parser service unreachable: {e}") from e

    if resp.status_code != 200:
        await ParserDashboardService.record_run(
            filename=filename, source="rag_ingest", status="failed", mode=mode,
            bytes_size=len(content), duration_ms=int((time.perf_counter() - t0) * 1000),
            error=f"parser service {resp.status_code}: {resp.text[:300]}",
        )
        raise ParserServiceError(f"parser service {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    duration_ms = int((time.perf_counter() - t0) * 1000)
    blocks = data.get("blocks") or []
    logger.info(
        "parser service ok",
        filename=filename,
        pages=data.get("pages"),
        blocks=len(blocks),
        figures=len(data.get("images") or []),
    )
    # status="empty" (not "failed") when the service responds 200 but finds
    # nothing to extract — the caller (ingestion_service) still treats this as
    # a fallback trigger, but it's a materially different outcome for the
    # dashboard than an unreachable/500 service.
    await ParserDashboardService.record_run(
        filename=filename, source="rag_ingest",
        status="success" if blocks else "empty",
        mode=mode, pages=data.get("pages"), blocks_count=len(blocks),
        figures_count=len(data.get("images") or []), bytes_size=len(content),
        duration_ms=duration_ms, markdown=data.get("markdown"),
    )
    return data


async def parser_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{PARSER_SERVICE_URL}/health",
                headers={"Authorization": f"Bearer {PARSER_SERVICE_KEY}"},
            )
            return r.status_code == 200
    except Exception:
        return False
