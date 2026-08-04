"""
Parser dashboard API — admin-only.

Serves the run history recorded by ParserDashboardService (every real call to
the custom AWS PDF-parser microservice, from both live RAG ingestion and the
/parser eval page) to the standalone dashboard page hosted directly on the
parser box (http://52.207.56.41/dashboard). That page calls these endpoints
cross-origin with the browser's existing admin session cookie
(credentials: 'include') — no separate login on the dashboard page itself.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId
from datetime import datetime

from routes.admin_routes import require_admin
from services.parser_dashboard_service import ParserDashboardService

router = APIRouter(prefix="/api/v1/parser-dashboard", tags=["Parser Dashboard"])


def _serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId / datetime fields to JSON-safe types."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


@router.get("/stats")
async def get_stats(_: dict = Depends(require_admin)):
    return await ParserDashboardService.get_stats()


@router.get("/runs")
async def list_runs(
    limit: int = Query(50, le=200),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    _: dict = Depends(require_admin),
):
    runs = await ParserDashboardService.list_runs(limit=limit, skip=skip, status=status)
    return {"runs": [_serialize(r) for r in runs]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, _: dict = Depends(require_admin)):
    run = await ParserDashboardService.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return _serialize(run)
