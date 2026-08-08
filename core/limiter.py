import os
import redis
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.auth import verify_token


def _user_or_ip_key(request: Request) -> str:
    """Rate-limit key: authenticated user_id when available, IP otherwise.

    IP-only keying let an authenticated attacker trivially bypass the limit
    on /chat/stream — an endpoint that already resolves current_user via
    get_current_user, and can spawn an up-to-150-step agentic background
    task per call — just by rotating source IPs. The auth layer that
    already exists wasn't being used for this.

    Decodes the JWT directly here (sync, no DB call) rather than depending
    on get_current_user having already run: slowapi's key_func is invoked
    before route dependencies resolve, so `request.state` doesn't have the
    resolved user yet at this point.
    """
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    if token:
        payload = verify_token(token)
        if payload and payload.get("user_id"):
            return f"user:{payload['user_id']}"
    return get_remote_address(request)


def _build_limiter():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            r = redis.Redis.from_url(redis_url, socket_timeout=1)
            r.ping()
            return Limiter(key_func=_user_or_ip_key, storage_uri=redis_url)
        except Exception as e:
            print(f"⚠️  Rate limiter Redis backend failed ({e}), using in-memory")
    return Limiter(key_func=_user_or_ip_key)

limiter = _build_limiter()
