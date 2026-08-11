"""
Client for the hybrid PDF parser microservice (DocLayout-YOLO + VLM, on the
parser box). The service is stateless: PDF in → typed blocks out. All
chunking, embedding and indexing stays here in the backend, so the parser
never needs database credentials or tenancy logic.
"""
import os
import logging
from typing import Any, Dict, Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://52.207.56.41/parser")
PARSER_SERVICE_KEY = os.getenv("PARSER_SERVICE_KEY", os.getenv("OMNIROUTE_API_KEY", ""))
# The parser's region-VLM calls run strictly sequentially by design (parallel
# calls trip the gateway's rate limits — see pdf-parser/parser.py's
# REGION_WORKERS comment), so a heavily-illustrated scan can legitimately take
# several minutes. But 1800s meant a user could sit waiting nearly 30 minutes
# on one upload before the LlamaParse fallback ever kicked in — confirmed
# live: a 15-page scanned book with ~100+ image regions ran into nginx's own
# 600s proxy_read_timeout on the parser box first anyway (a 504, unrelated to
# this value), wasting 10 minutes before falling back to a LlamaParse pass
# that itself took under a minute. Bounded to a more interactive-upload-sized
# window; nginx's proxy_read_timeout on the parser box is set to match (see
# TIER2_SANDBOX_PLAN.md / the parser box's /etc/nginx/sites-available/omniroute).
PARSER_TIMEOUT = float(os.getenv("PARSER_TIMEOUT", "300"))


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
        raise ParserServiceError(f"parser service unreachable: {e}") from e

    if resp.status_code != 200:
        raise ParserServiceError(f"parser service {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    logger.info(
        "parser service ok",
        filename=filename,
        pages=data.get("pages"),
        blocks=len(data.get("blocks") or []),
        figures=len(data.get("images") or []),
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
