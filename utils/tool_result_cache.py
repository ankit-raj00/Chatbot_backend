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
    "web_search":                 300,  # 5 min (TinyFish)
    "search_knowledge_base":      120,  # 2 min
    "list_google_drive_folders":   60,  # 1 min
}
# NOTE: get_current_time is intentionally NOT cached — deduping a clock reading
# is pointless and would return a stale timestamp. Side-effecting tools
# (run_python/run_shell/document generation) are excluded by omission.


def _cache_key(name: str, args: dict) -> str:
    # Include user_id explicitly (also present in args today) so a cached result
    # can never leak across users, even if a future tool derives it differently.
    user_scope = str(args.get("user_id", "")) if isinstance(args, dict) else ""
    h = hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"tool_cache:{name}:{user_scope}:{h}"


async def cached_invoke(
    tool_name: str,
    tool_args: dict,
    execute_fn: Callable[[], Awaitable[Any]],
) -> Any:
    """
    Execute a tool with Redis caching. Falls through to execute_fn on cache miss.
    Non-cacheable tools always execute immediately. The tool is executed AT MOST
    ONCE even if the cache backend errors.
    """
    ttl = CACHEABLE.get(tool_name)
    if ttl is None:
        return await execute_fn()   # not cacheable — always run

    key = _cache_key(tool_name, tool_args)

    # ── Try cache read (non-fatal on error) ──────────────────────
    try:
        from core.cache import cache_get
        hit = await cache_get(key)
        if hit is not None:
            logger.debug(f"tool_cache HIT: {tool_name}")
            return hit
    except Exception as e:
        logger.warning(f"tool_cache read error for {tool_name}: {e}")

    # ── Execute exactly once ─────────────────────────────────────
    result = await execute_fn()

    # ── Try cache write (non-fatal on error) ─────────────────────
    try:
        from core.cache import cache_set
        await cache_set(key, result, ttl_seconds=ttl)
    except Exception as e:
        logger.warning(f"tool_cache write error for {tool_name}: {e}")

    return result
