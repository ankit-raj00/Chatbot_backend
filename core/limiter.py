import os
import redis
from slowapi import Limiter
from slowapi.util import get_remote_address

def _build_limiter():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            r = redis.Redis.from_url(redis_url, socket_timeout=1)
            r.ping()
            return Limiter(key_func=get_remote_address, storage_uri=redis_url)
        except Exception as e:
            print(f"⚠️  Rate limiter Redis backend failed ({e}), using in-memory")
    return Limiter(key_func=get_remote_address)

limiter = _build_limiter()
