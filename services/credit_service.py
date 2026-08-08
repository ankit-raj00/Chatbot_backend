"""
CreditService — per-user spend tracking against a free-tier credit cap.

Mongo (users_collection) is the source of truth: `credits_used_usd` (running
total spent) and `credit_cap_usd` (defaults to DEFAULT_CREDIT_CAP_USD if
absent — deliberately per-user so a future paid tier can just set a higher
cap on that user's doc, no schema migration needed). Redis is a read-through
cache to avoid a Mongo round trip on the hot pre-turn check.

Mirrors services/ingestion_job_service.py's pattern: prefix constant, TTL'd
Redis keys, best-effort error handling that never breaks the caller.
"""
import os
from typing import Optional

from bson import ObjectId

from core.cache import cache_get, cache_set, get_redis
from core.database import users_collection

import structlog
logger = structlog.get_logger(__name__)

CREDIT_KEY_PREFIX = "credit_used"
DEFAULT_CREDIT_CAP_USD = float(os.getenv("DEFAULT_CREDIT_CAP_USD", "5.0"))
CREDIT_GRACE_USD = float(os.getenv("CREDIT_GRACE_USD", "1.0"))
CREDIT_CACHE_TTL_SECONDS = 3600  # 1 hour — spend is re-synced from Mongo on miss/expiry


class CreditService:

    @staticmethod
    def _key(user_id: str) -> str:
        return f"{CREDIT_KEY_PREFIX}:{user_id}"

    @classmethod
    async def get_spend(cls, user_id: str) -> float:
        """Redis fast path; on miss, read from Mongo (source of truth) and
        repopulate the cache."""
        cached = await cache_get(cls._key(user_id))
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass  # fall through to Mongo on a corrupt cache value

        user = await users_collection.find_one({"_id": ObjectId(user_id)}, {"credits_used_usd": 1})
        spend = float((user or {}).get("credits_used_usd", 0.0))
        try:
            await cache_set(cls._key(user_id), spend, ttl_seconds=CREDIT_CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to cache credit spend for user {user_id}: {e}")
        return spend

    @staticmethod
    async def get_cap(user_id: str) -> float:
        user = await users_collection.find_one({"_id": ObjectId(user_id)}, {"credit_cap_usd": 1})
        cap = (user or {}).get("credit_cap_usd")
        return float(cap) if cap is not None else DEFAULT_CREDIT_CAP_USD

    @staticmethod
    async def _is_admin(user_id: str) -> bool:
        user = await users_collection.find_one({"_id": ObjectId(user_id)}, {"is_admin": 1})
        return bool((user or {}).get("is_admin", False))

    @classmethod
    async def has_credit(cls, user_id: str) -> bool:
        # Admins are exempt from the block (so the person running/demoing
        # this app doesn't get rate-limited by its own free-tier logic) but
        # their spend is still tracked exactly like everyone else's via
        # record_and_deduct — this only skips enforcement, not accounting.
        if await cls._is_admin(user_id):
            return True
        spend, cap = await cls.get_spend(user_id), await cls.get_cap(user_id)
        return spend < cap

    @classmethod
    async def record_and_deduct(cls, user_id: str, cost_usd: float) -> None:
        """Called after a turn completes. Mongo `$inc` is the durable write
        (atomic under concurrent turns, and `$inc` on a missing field just
        creates it — no migration needed for existing users). The Redis
        increment is best-effort only; losing it just means the next
        get_spend() falls back to Mongo and re-syncs the cache."""
        if cost_usd <= 0:
            return
        try:
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$inc": {"credits_used_usd": cost_usd}},
            )
        except Exception as e:
            logger.error(f"Failed to deduct credit for user {user_id}: {e}")
            return  # don't touch the cache if the durable write itself failed

        try:
            r = await get_redis()
            if r is not None:
                key = cls._key(user_id)
                await r.incrbyfloat(key, cost_usd)
                await r.expire(key, CREDIT_CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to update Redis credit cache for user {user_id}: {e}")
