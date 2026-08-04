"""
ParserDashboardService — records every real call to the custom AWS PDF-parser
microservice (both from live RAG ingestion and the /parser eval page) and
serves that history back to the admin-only parser dashboard.

The parser microservice itself is deliberately stateless and has no DB
credentials (see rag/parsers/parser_client.py) — this backend is the one
place that already has Mongo, so it's the one place that should own
persistence for "what has the parser done so far."
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from core.database import parser_runs_collection

import structlog
logger = structlog.get_logger(__name__)

MAX_RUNS_PER_PAGE = 200


class ParserDashboardService:

    @staticmethod
    async def record_run(
        *,
        filename: str,
        source: str,                       # "rag_ingest" | "eval_page"
        status: str,                       # "success" | "empty" | "failed"
        mode: str = "auto",
        pages: Optional[int] = None,
        blocks_count: Optional[int] = None,
        figures_count: Optional[int] = None,
        bytes_size: Optional[int] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        markdown: Optional[str] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """Best-effort — a Mongo hiccup here must never break a real parse."""
        try:
            await parser_runs_collection.insert_one({
                "run_id": str(uuid.uuid4()),
                "filename": filename,
                "source": source,
                "status": status,
                "mode": mode,
                "pages": pages,
                "blocks_count": blocks_count,
                "figures_count": figures_count,
                "bytes": bytes_size,
                "duration_ms": duration_ms,
                "error": error,
                "markdown": markdown,
                "user_id": user_id,
                "user_email": user_email,
                "created_at": datetime.now(timezone.utc),
            })
        except Exception as e:
            logger.warning(f"Failed to record parser run for {filename}: {e}")

    @staticmethod
    async def get_stats() -> dict:
        total = await parser_runs_collection.count_documents({})
        success = await parser_runs_collection.count_documents({"status": "success"})
        empty = await parser_runs_collection.count_documents({"status": "empty"})
        failed = await parser_runs_collection.count_documents({"status": "failed"})

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
        agg_result = await parser_runs_collection.aggregate(pipeline).to_list(length=1)
        agg = agg_result[0] if agg_result else {}

        return {
            "total_runs": total,
            "success": success,
            "empty": empty,
            "failed": failed,
            "avg_duration_ms": round(agg.get("avg_duration_ms") or 0),
            "total_pages": agg.get("total_pages") or 0,
            "total_blocks": agg.get("total_blocks") or 0,
            "total_figures": agg.get("total_figures") or 0,
        }

    @staticmethod
    async def list_runs(limit: int = 50, skip: int = 0, status: Optional[str] = None) -> list[dict]:
        query = {"status": status} if status else {}
        # Markdown excluded here — can be large and isn't needed until a run is opened.
        cursor = (
            parser_runs_collection.find(query, {"markdown": 0})
            .sort("created_at", -1)
            .skip(skip)
            .limit(min(limit, MAX_RUNS_PER_PAGE))
        )
        return await cursor.to_list(length=limit)

    @staticmethod
    async def get_run(run_id: str) -> Optional[dict]:
        return await parser_runs_collection.find_one({"run_id": run_id})
