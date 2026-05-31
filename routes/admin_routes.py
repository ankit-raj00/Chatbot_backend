"""
Admin Analytics Routes
Protected by is_admin flag. Provides per-user, per-session, per-turn cost analytics.

Promote a user to admin via MongoDB:
    db.users.updateOne({ email: "your@email.com" }, { $set: { is_admin: true } })
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from core.middleware import get_current_user
from core.database import users_collection, messages_collection, conversations_collection
from bson import ObjectId
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

# ─── Admin guard dependency ───────────────────────────────────────────────────

async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency that raises 403 if the user is not an admin."""
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )
    return current_user


def _serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId / datetime fields to JSON-safe types."""
    result = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _serialize(v)
        elif isinstance(v, list):
            result[k] = [_serialize(i) if isinstance(i, dict) else (str(i) if isinstance(i, ObjectId) else i) for i in v]
        else:
            result[k] = v
    return result


# ─── Overview ─────────────────────────────────────────────────────────────────

@router.get("/overview")
async def get_overview(_: dict = Depends(require_admin)):
    """Top-level platform stats."""
    total_users = await users_collection.count_documents({})
    total_conversations = await conversations_collection.count_documents({})
    total_messages = await messages_collection.count_documents({})

    # Today's message count
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    messages_today = await messages_collection.count_documents({"timestamp": {"$gte": today_start}})

    # Aggregate total tokens & cost from AI (model) messages
    pipeline = [
        {"$match": {"role": "model"}},
        {"$group": {
            "_id": None,
            "total_input_tokens":  {"$sum": "$input_tokens"},
            "total_output_tokens": {"$sum": "$output_tokens"},
            "total_cost_usd":      {"$sum": "$cost_usd"},
        }}
    ]
    agg = await messages_collection.aggregate(pipeline).to_list(1)
    tokens_data = agg[0] if agg else {}

    return {
        "total_users":         total_users,
        "total_conversations": total_conversations,
        "total_messages":      total_messages,
        "messages_today":      messages_today,
        "total_input_tokens":  tokens_data.get("total_input_tokens", 0),
        "total_output_tokens": tokens_data.get("total_output_tokens", 0),
        "total_cost_usd":      round(tokens_data.get("total_cost_usd", 0.0), 6),
    }


# ─── All Users with aggregate stats ──────────────────────────────────────────

@router.get("/users")
async def get_all_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    _: dict = Depends(require_admin)
):
    """Paginated list of all users with per-user usage stats."""
    skip = (page - 1) * limit
    users_cursor = users_collection.find(
        {}, {"password": 0}
    ).sort("created_at", -1).skip(skip).limit(limit)
    users = await users_cursor.to_list(limit)

    result = []
    for user in users:
        uid = str(user["_id"])
        # Per-user aggregates from model messages
        pipeline = [
            {"$match": {"user_id": uid, "role": "model"}},
            {"$group": {
                "_id": None,
                "total_input_tokens":  {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "total_cost_usd":      {"$sum": "$cost_usd"},
                "message_count":       {"$sum": 1},
            }}
        ]
        agg = await messages_collection.aggregate(pipeline).to_list(1)
        stats = agg[0] if agg else {}

        conv_count = await conversations_collection.count_documents({"user_id": uid})

        result.append({
            "id":                  uid,
            "name":                user.get("name", ""),
            "email":               user.get("email", ""),
            "is_admin":            user.get("is_admin", False),
            "created_at":          user["created_at"].isoformat() if isinstance(user.get("created_at"), datetime) else str(user.get("created_at", "")),
            "conversations":       conv_count,
            "ai_messages":         stats.get("message_count", 0),
            "total_input_tokens":  stats.get("total_input_tokens", 0),
            "total_output_tokens": stats.get("total_output_tokens", 0),
            "total_cost_usd":      round(stats.get("total_cost_usd", 0.0), 6),
        })

    total = await users_collection.count_documents({})
    return {"users": result, "total": total, "page": page, "limit": limit}


# ─── Sessions for a specific user ─────────────────────────────────────────────

@router.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, _: dict = Depends(require_admin)):
    """All conversations for a user with per-session stats."""
    convs = await conversations_collection.find(
        {"user_id": user_id}
    ).sort("updated_at", -1).to_list(200)

    result = []
    for conv in convs:
        conv_id = str(conv["_id"])
        # Count messages and aggregate cost for this session
        pipeline = [
            {"$match": {"conversation_id": conv_id, "role": "model"}},
            {"$group": {
                "_id": None,
                "total_cost_usd":      {"$sum": "$cost_usd"},
                "total_input_tokens":  {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "turns":               {"$sum": 1},
            }}
        ]
        agg = await messages_collection.aggregate(pipeline).to_list(1)
        stats = agg[0] if agg else {}
        total_msgs = await messages_collection.count_documents({"conversation_id": conv_id})

        result.append({
            "id":                  conv_id,
            "title":               conv.get("title", "Untitled"),
            "created_at":          conv["created_at"].isoformat() if isinstance(conv.get("created_at"), datetime) else "",
            "updated_at":          conv["updated_at"].isoformat() if isinstance(conv.get("updated_at"), datetime) else "",
            "total_messages":      total_msgs,
            "ai_turns":            stats.get("turns", 0),
            "total_input_tokens":  stats.get("total_input_tokens", 0),
            "total_output_tokens": stats.get("total_output_tokens", 0),
            "total_cost_usd":      round(stats.get("total_cost_usd", 0.0), 6),
        })

    return {"sessions": result, "count": len(result)}


# ─── Turns inside a session ───────────────────────────────────────────────────

@router.get("/users/{user_id}/sessions/{conv_id}")
async def get_session_turns(user_id: str, conv_id: str, _: dict = Depends(require_admin)):
    """All messages (turns) in a conversation with per-turn cost."""
    msgs = await messages_collection.find(
        {"conversation_id": conv_id, "user_id": user_id}
    ).sort("timestamp", 1).to_list(1000)

    return {
        "turns": [_serialize(m) for m in msgs],
        "count": len(msgs)
    }


# ─── Daily usage (last 30 days) ───────────────────────────────────────────────

@router.get("/usage/daily")
async def get_daily_usage(_: dict = Depends(require_admin)):
    """Message volume per day for the last 30 days."""
    since = datetime.now() - timedelta(days=30)
    pipeline = [
        {"$match": {"timestamp": {"$gte": since}, "role": "model"}},
        {"$group": {
            "_id": {
                "year":  {"$year": "$timestamp"},
                "month": {"$month": "$timestamp"},
                "day":   {"$dayOfMonth": "$timestamp"},
            },
            "messages":    {"$sum": 1},
            "cost_usd":    {"$sum": "$cost_usd"},
            "input_tokens":  {"$sum": "$input_tokens"},
            "output_tokens": {"$sum": "$output_tokens"},
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}}
    ]
    raw = await messages_collection.aggregate(pipeline).to_list(31)
    result = []
    for r in raw:
        d = r["_id"]
        result.append({
            "date":          f"{d['year']}-{d['month']:02d}-{d['day']:02d}",
            "messages":      r["messages"],
            "cost_usd":      round(r.get("cost_usd", 0.0), 6),
            "input_tokens":  r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
        })
    return {"daily": result}


# ─── Model usage breakdown ────────────────────────────────────────────────────

@router.get("/usage/models")
async def get_model_usage(_: dict = Depends(require_admin)):
    """Which models are used most."""
    pipeline = [
        {"$match": {"role": "model", "model": {"$exists": True}}},
        {"$group": {
            "_id":           "$model",
            "count":         {"$sum": 1},
            "total_cost":    {"$sum": "$cost_usd"},
            "total_input":   {"$sum": "$input_tokens"},
            "total_output":  {"$sum": "$output_tokens"},
        }},
        {"$sort": {"count": -1}}
    ]
    raw = await messages_collection.aggregate(pipeline).to_list(20)
    return {"models": [{"model": r["_id"], "count": r["count"], "total_cost_usd": round(r.get("total_cost", 0.0), 6), "total_input_tokens": r.get("total_input", 0), "total_output_tokens": r.get("total_output", 0)} for r in raw]}


# ─── Tool usage breakdown ─────────────────────────────────────────────────────

@router.get("/usage/tools")
async def get_tool_usage(_: dict = Depends(require_admin)):
    """Which tools are called most."""
    pipeline = [
        {"$match": {"role": "model", "tool_steps": {"$exists": True, "$ne": []}}},
        {"$unwind": "$tool_steps"},
        {"$group": {
            "_id":   "$tool_steps.name",
            "count": {"$sum": 1},
        }},
        {"$sort": {"count": -1}},
        {"$limit": 20}
    ]
    raw = await messages_collection.aggregate(pipeline).to_list(20)
    return {"tools": [{"tool": r["_id"], "count": r["count"]} for r in raw]}
