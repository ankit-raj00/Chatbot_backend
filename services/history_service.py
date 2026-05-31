"""
HistoryService — loads conversation history from MongoDB with Redis caching.

Cache key format : history:{conversation_id}
Cache TTL        : 1800 seconds (30 minutes)
Cache invalidation: called by ChatService after saving the AI response

WHY 30-minute TTL: Active conversations are cached; idle ones expire naturally.
WHY cache-then-DB: MongoDB round-trip is ~20-100ms. For a streaming chat,
                   loading history before each turn adds measurable latency.
"""

import json
import logging
from typing import Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core.database import messages_collection
from core.cache import cache_get, cache_set, cache_delete

import structlog
logger = structlog.get_logger(__name__)

HISTORY_CACHE_TTL = 1800   # 30 minutes
HISTORY_CACHE_PREFIX = "history"
MAX_HISTORY_MESSAGES = 30  # Keep last 30 messages for context


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain message objects to JSON-serializable dicts."""
    serialized = []
    for msg in messages:
        entry = {
            "role": "human" if isinstance(msg, HumanMessage) else "ai",
        }
        # Content can be str or list of content parts (multimodal)
        if isinstance(msg.content, str):
            entry["content"] = msg.content
        elif isinstance(msg.content, list):
            entry["content"] = msg.content   # store as-is (already JSON-serializable)
        serialized.append(entry)
    return serialized


def _deserialize_messages(serialized: list[dict]) -> list[BaseMessage]:
    """Rebuild LangChain message objects from cached dicts."""
    messages = []
    for entry in serialized:
        content = entry["content"]
        if entry["role"] == "human":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


class HistoryService:

    @staticmethod
    async def get_history(
        conversation_id: str,
        user_id: str,
        exclude_msg_id=None        # ObjectId of the just-inserted user message
    ) -> list[BaseMessage]:
        """
        Returns conversation history as LangChain message objects.
        Tries Redis first, falls back to MongoDB.
        """
        cache_key = f"{HISTORY_CACHE_PREFIX}:{conversation_id}"

        # ── Try Redis cache ──────────────────────────────────
        try:
            cached = await cache_get(cache_key)
            if cached and isinstance(cached, list):
                logger.debug(f"History cache HIT for {conversation_id}")
                return _deserialize_messages(cached)
        except Exception as e:
            logger.warning(f"Redis cache read failed (falling back to DB): {e}")

        # ── Cache miss: load from MongoDB ────────────────────
        logger.debug(f"History cache MISS for {conversation_id}, loading from MongoDB")

        query = {
            "conversation_id": conversation_id,
            "user_id": user_id,
        }
        if exclude_msg_id is not None:
            query["_id"] = {"$ne": exclude_msg_id}

        cursor = messages_collection.find(query).sort("timestamp", 1)
        stored = await cursor.to_list(length=MAX_HISTORY_MESSAGES)

        messages = []
        for msg in stored:
            content_parts = [{"type": "text", "text": msg.get("content", "")}]
            if msg.get("attachments"):
                for att in msg["attachments"]:
                    uri = att.get("gemini_uri") or att.get("uri")
                    if uri:
                        content_parts.append({
                            "type": "file",
                            "file_id": uri,
                            "mime_type": att.get("mime_type")
                        })
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content_parts if len(content_parts) > 1 else msg.get("content", "")))
            else:
                messages.append(AIMessage(content=msg.get("content", "")))

        # ── Write to Redis for next request ──────────────────
        try:
            await cache_set(cache_key, _serialize_messages(messages), ttl_seconds=HISTORY_CACHE_TTL)
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")

        return messages

    @staticmethod
    async def invalidate(conversation_id: str) -> None:
        """
        Called after saving the AI response to force a fresh DB read next turn.
        Without this, the cached history won't include the just-saved AI message.
        """
        try:
            await cache_delete(f"{HISTORY_CACHE_PREFIX}:{conversation_id}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed for {conversation_id}: {e}")
