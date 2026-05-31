"""
Redis async client — singleton connection pool.
Used for: conversation history cache, MCP tool list cache,
          ingestion job status, OAuth state, rate limiting.

Import pattern in other modules:
    from core.cache import get_redis
    r = await get_redis()
    await r.set("key", "value", ex=300)
    value = await r.get("key")
"""

import os
import json
import logging
from typing import Any, Optional
import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool, Redis

logger = logging.getLogger(__name__)

# Module-level pool — created once on startup, shared across all requests
_pool: Optional[ConnectionPool] = None
_client: Optional[Redis] = None
_fallback_cache: dict = {}


async def init_redis() -> None:
    """
    Initialize the Redis connection pool.
    Called from main.py lifespan on startup.
    Raises RuntimeError if connection cannot be established.
    """
    global _pool, _client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    _pool = ConnectionPool.from_url(
        redis_url,
        max_connections=20,          # max 20 simultaneous connections
        decode_responses=True,       # return str, not bytes
        socket_connect_timeout=5,    # fail fast if Redis is unreachable
        socket_timeout=5,
        retry_on_timeout=True,
    )
    _client = Redis(connection_pool=_pool)

    # Verify the connection actually works
    try:
        await _client.ping()
        logger.info(f"Redis connected: {redis_url}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        _client = None
        _pool = None
        raise RuntimeError(
            f"Cannot connect to Redis at {redis_url}. "
            "Set REDIS_URL in your .env file. "
            "For local dev: run `docker run -p 6379:6379 redis:alpine`"
        ) from e


async def close_redis() -> None:
    """
    Close the Redis connection pool.
    Called from main.py lifespan on shutdown.
    """
    global _client, _pool
    if _client:
        await _client.aclose()
    if _pool:
        await _pool.aclose()
    logger.info("Redis connection closed")


async def get_redis() -> Optional[Redis]:
    """
    Returns the shared Redis client or None if not initialized.
    """
    return _client


# ─── Convenience helpers ──────────────────────────────────────────────────────

async def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Serialize value to JSON and store with TTL or fallback to memory."""
    if _client is None:
        _fallback_cache[key] = json.dumps(value)
        return
    try:
        await _client.set(key, json.dumps(value), ex=ttl_seconds)
    except Exception as e:
        logger.warning(f"Redis set failed, using fallback: {e}")
        _fallback_cache[key] = json.dumps(value)


async def cache_get(key: str) -> Optional[Any]:
    """Retrieve and deserialize a JSON value from Redis or fallback. Returns None if not found."""
    raw = None
    if _client is None:
        raw = _fallback_cache.get(key)
    else:
        try:
            raw = await _client.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed, using fallback: {e}")
            raw = _fallback_cache.get(key)
            
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw   # return raw string if not valid JSON

async def cache_delete(key: str) -> None:
    """Delete a key from Redis or fallback cache."""
    if _client is None:
        _fallback_cache.pop(key, None)
        return
    try:
        await _client.delete(key)
    except Exception as e:
        logger.warning(f"Redis delete failed, using fallback: {e}")
        _fallback_cache.pop(key, None)


async def cache_exists(key: str) -> bool:
    """Returns True if key exists in Redis or fallback cache."""
    if _client is None:
        return key in _fallback_cache
    try:
        return await _client.exists(key) > 0
    except Exception as e:
        logger.warning(f"Redis exists failed, using fallback: {e}")
        return key in _fallback_cache
