"""
Run history for the parser dashboard — Mongo-backed, but deliberately owned
by THIS service, not borrowed from any caller's database. The parser stays
a standalone product this way: whoever calls POST /parse (the chatbot's RAG
ingestion, its eval page, or a future unrelated customer) gets recorded the
same way, and this service's own persistence doesn't depend on anyone else's
backend being reachable or their schema staying stable.

Currently points at the same Mongo Atlas cluster the chatbot uses (free,
already provisioned, zero new infra) but a dedicated database — MONGO_URI /
MONGO_DB_NAME are just this service's own env vars, swappable to a different
cluster or provider later with no code change.
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "parser_dashboard")
MAX_RUNS_PER_PAGE = 200

_client: Optional[AsyncIOMotorClient] = None
_runs = None
_pages = None


def init_db() -> None:
    """Call once at app startup. No-ops (dashboard just stays empty) if
    MONGO_URI isn't set, rather than crashing the whole parser service over
    a dashboard-only dependency."""
    global _client, _runs, _pages
    if not MONGO_URI:
        return
    _client = AsyncIOMotorClient(MONGO_URI)
    db = _client[MONGO_DB_NAME]
    _runs = db["runs"]
    # Separate collection, not embedded in the run doc — a page image is
    # ~100-500KB base64, and a many-page document embedded in one Mongo doc
    # would risk the 16MB single-document limit. One doc per page instead.
    _pages = db["pages"]


async def record_run(
    *,
    filename: str,
    status: str,                       # "success" | "empty" | "failed"
    mode: str = "auto",
    pages: Optional[int] = None,
    blocks_count: Optional[int] = None,
    figures_count: Optional[int] = None,
    bytes_size: Optional[int] = None,
    duration_ms: Optional[int] = None,
    error: Optional[str] = None,
    markdown: Optional[str] = None,
) -> str:
    """Best-effort — a Mongo hiccup here must never break a real parse.
    Returns the run_id regardless (even on insert failure) so the caller can
    still try attaching page previews under it."""
    run_id = str(uuid.uuid4())
    if _runs is None:
        return run_id
    try:
        await _runs.insert_one({
            "run_id": run_id,
            "filename": filename,
            "status": status,
            "mode": mode,
            "pages": pages,
            "blocks_count": blocks_count,
            "figures_count": figures_count,
            "bytes": bytes_size,
            "duration_ms": duration_ms,
            "error": error,
            "markdown": markdown,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass
    return run_id


async def record_pages(run_id: str, pages: list[dict]) -> None:
    """Per-page previews (rendered page image URL + exactly what was
    extracted from that specific page) — lets the dashboard show a real
    side-by-side compare instead of just the whole document's markdown."""
    if _pages is None or not pages:
        return
    try:
        await _pages.insert_many([{"run_id": run_id, **p} for p in pages])
    except Exception:
        pass


async def list_pages(run_id: str) -> list[dict]:
    if _pages is None:
        return []
    cursor = _pages.find({"run_id": run_id}, {"_id": 0}).sort("page", 1)
    return await cursor.to_list(length=500)


async def get_stats() -> dict:
    empty_stats = {
        "total_runs": 0, "success": 0, "empty": 0, "failed": 0,
        "avg_duration_ms": 0, "total_pages": 0, "total_blocks": 0, "total_figures": 0,
    }
    if _runs is None:
        return empty_stats

    total = await _runs.count_documents({})
    success = await _runs.count_documents({"status": "success"})
    empty = await _runs.count_documents({"status": "empty"})
    failed = await _runs.count_documents({"status": "failed"})

    pipeline = [
        {"$match": {"status": {"$in": ["success", "empty"]}}},
        {"$group": {
            "_id": None,
            "avg_duration_ms": {"$avg": "$duration_ms"},
            "total_pages": {"$sum": "$pages"},
            "total_blocks": {"$sum": "$blocks_count"},
            "total_figures": {"$sum": "$figures_count"},
        }},
    ]
    agg_result = await _runs.aggregate(pipeline).to_list(length=1)
    agg = agg_result[0] if agg_result else {}

    return {
        "total_runs": total, "success": success, "empty": empty, "failed": failed,
        "avg_duration_ms": round(agg.get("avg_duration_ms") or 0),
        "total_pages": agg.get("total_pages") or 0,
        "total_blocks": agg.get("total_blocks") or 0,
        "total_figures": agg.get("total_figures") or 0,
    }


async def list_runs(limit: int = 50, skip: int = 0, status: Optional[str] = None) -> list[dict]:
    if _runs is None:
        return []
    query = {"status": status} if status else {}
    # _id/markdown excluded — _id isn't JSON-serializable as-is, markdown can
    # be large and isn't needed until a single run is opened.
    cursor = (
        _runs.find(query, {"_id": 0, "markdown": 0})
        .sort("created_at", -1)
        .skip(skip)
        .limit(min(limit, MAX_RUNS_PER_PAGE))
    )
    return await cursor.to_list(length=limit)


async def get_run(run_id: str) -> Optional[dict]:
    if _runs is None:
        return None
    return await _runs.find_one({"run_id": run_id}, {"_id": 0})
