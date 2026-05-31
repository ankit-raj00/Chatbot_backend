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


async def get_redis() -> Redis:
    """
    Returns the shared Redis client.
    Raises RuntimeError if init_redis() was never called.
    """
    if _client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() at startup.")
    return _client


# ─── Convenience helpers ──────────────────────────────────────────────────────

async def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Serialize value to JSON and store with TTL."""
    r = await get_redis()
    await r.set(key, json.dumps(value), ex=ttl_seconds)


async def cache_get(key: str) -> Optional[Any]:
    """Retrieve and deserialize a JSON value. Returns None if not found."""
    r = await get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw   # return raw string if not valid JSON


async def cache_delete(key: str) -> None:
    """Delete a cache key."""
    r = await get_redis()
    await r.delete(key)


async def cache_exists(key: str) -> bool:
    """Returns True if key exists in Redis."""
    r = await get_redis()
    return await r.exists(key) > 0
