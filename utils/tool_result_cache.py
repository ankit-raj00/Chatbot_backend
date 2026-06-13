"""
Tool result cache — Redis-backed, deduplicated tool call results.

Only caches idempotent tools (same input → same output within TTL).
Never caches: shell commands (side effects), document generation, write operations.

Key format: tool_cache:{tool_name}:{md5(sorted_args)}
"""

import hashlib
import json
import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

CACHEABLE: dict[str, int] = {          # tool_name → TTL seconds
    "get_weather":                600,  # 10 min
    "get_current_time":            30,  # 30 sec (dedup only)
    "tavily_search":              300,  # 5 min
    "search_knowledge_base":      120,  # 2 min
    "list_google_drive_folders":   60,  # 1 min
}


def _cache_key(name: str, args: dict) -> str:
    h = hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"tool_cache:{name}:{h}"


async def cached_invoke(
    tool_name: str,
    tool_args: dict,
    execute_fn: Callable[[], Awaitable[Any]],
) -> Any:
    """
    Execute a tool with Redis caching. Falls through to execute_fn on cache miss.
    Non-cacheable tools always execute immediately.
    """
    ttl = CACHEABLE.get(tool_name)
    if ttl is None:
        return await execute_fn()   # not cacheable — always run

    try:
        from core.cache import cache_get, cache_set
        key = _cache_key(tool_name, tool_args)
        hit = await cache_get(key)
        if hit is not None:
            logger.debug(f"tool_cache HIT: {tool_name}")
            return hit

        result = await execute_fn()
        await cache_set(key, result, ttl_seconds=ttl)
        return result
    except Exception as e:
        # Cache failure is non-fatal — run tool directly
        logger.warning(f"tool_cache error for {tool_name}: {e}")
        return await execute_fn()
