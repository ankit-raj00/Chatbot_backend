"""
CreditService — per-user spend tracking against a free-tier credit cap.

Mongo (users_collection) is the source of truth: `credits_used_usd` (running
total spent) and `credit_cap_usd` (defaults to DEFAULT_CREDIT_CAP_USD if
absent — deliberately per-user so a future paid tier can just set a higher
cap on that user's doc, no schema migration needed). Redis is a read-through
cache to avoid a Mongo round trip on the hot pre-turn check.

Mirrors services/ingestion_job_service.py's pattern: prefix constant, TTL'd
Redis keys, best-effort error handling that never breaks the caller.

SECURITY NOTE — concurrent-turn race: the pre-turn check (has_credit) and
the deduction (record_and_deduct, which only happens after a turn fully
completes) are NOT atomic with each other. Without acquire_turn_lock/
release_turn_lock below, a user could fire N concurrent turns (e.g. across
multiple conversations, or directly against the API bypassing the
frontend's single-composer UX) and every single one would pass has_credit()
independently, since none of them have deducted anything yet — the cap
would only actually apply to the (N+1)th turn, not the Nth. acquire_turn_lock
closes this by allowing only one in-flight turn per user at a time, checked
atomically via Redis SET NX.
"""
import os
import asyncio
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

TURN_LOCK_PREFIX = "active_turn_lock"
# Safety-net expiry, not the expected lifetime — release_turn_lock() should
# always fire first (chat_service.py releases it in _run_turn's finally
# block, on every exit path). This just self-heals if the process died
# between acquiring the lock and releasing it, rather than locking a user
# out of chat forever. Well above the longest realistic turn (recursion
# limit 150 tool-call round trips).
TURN_LOCK_TTL_SECONDS = 900


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

    @staticmethod
    def _lock_key(user_id: str) -> str:
        return f"{TURN_LOCK_PREFIX}:{user_id}"

    @classmethod
    async def acquire_turn_lock(cls, user_id: str) -> bool:
        """Atomic SET NX — only one in-flight turn per user at a time, so
        has_credit()'s pre-check can't be raced by starting several turns
        before any of them has deducted (see the module docstring). Returns
        False if the user already has an active turn (caller should reject
        the new one instead of starting it).

        Fails OPEN (allows the turn) if Redis itself is unreachable, matching
        this app's existing "Redis is optional, degrade gracefully" posture
        (core/cache.py) — has_credit()'s own Mongo fallback still enforces
        the actual spend cap even if this concurrency guard is temporarily
        unavailable; losing double-submission protection during a Redis
        outage is an acceptable tradeoff against taking all of chat down
        with it.
        """
        try:
            r = await get_redis()
            if r is None:
                return True
            return bool(await r.set(cls._lock_key(user_id), "1", nx=True, ex=TURN_LOCK_TTL_SECONDS))
        except Exception as e:
            logger.warning(f"Failed to acquire turn lock for user {user_id}: {e}")
            return True

    @classmethod
    async def release_turn_lock(cls, user_id: str) -> None:
        try:
            r = await get_redis()
            if r is not None:
                await r.delete(cls._lock_key(user_id))
        except Exception as e:
            logger.warning(f"Failed to release turn lock for user {user_id}: {e}")

    @classmethod
    async def record_and_deduct(cls, user_id: str, cost_usd: float, max_retries: int = 3) -> None:
        """Called after a turn completes (including a stopped/cancelled one
        — see chat_service.py's _persist_stopped_message). Mongo `$inc` is
        the durable write (atomic under concurrent turns, and `$inc` on a
        missing field just creates it — no migration needed for existing
        users). Retried a few times with backoff: this runs as a detached
        background task (spawn()), so a transient Mongo blip with no retry
        would silently and permanently lose that turn's cost — the user
        keeps the credit they should have been charged, with nothing left
        to reconcile it later. The Redis increment stays best-effort only;
        losing it just means the next get_spend() falls back to Mongo and
        re-syncs the cache."""
        if cost_usd <= 0:
            return

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                await users_collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$inc": {"credits_used_usd": cost_usd}},
                )
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * attempt)  # 0.5s, 1.0s

        if last_exc is not None:
            logger.error(
                f"Failed to deduct credit for user {user_id} after {max_retries} attempts "
                f"(cost_usd={cost_usd} was never recorded): {last_exc}"
            )
            return  # don't touch the cache if the durable write itself never succeeded

        try:
            r = await get_redis()
            if r is not None:
                key = cls._key(user_id)
                await r.incrbyfloat(key, cost_usd)
                await r.expire(key, CREDIT_CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to update Redis credit cache for user {user_id}: {e}")
