"""
Circuit Breaker — prevents cascading failures when external services fail.

States:
  CLOSED    → normal, requests pass through
  OPEN      → failing fast, all requests rejected immediately
  HALF_OPEN → recovery test, one request allowed through

Without this: 100 concurrent requests all wait through 30s backoff = system hangs.
With this: after N failures, all requests fail immediately with a clear message.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable, Awaitable
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class ServiceUnavailableError(Exception):
    """Raised when a circuit is OPEN and cannot accept requests."""
    pass


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self._state            = CircuitState.CLOSED
        self._failures         = 0
        self._last_fail: float = 0.0
        self._lock             = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _should_recover(self) -> bool:
        return (
            self._state == CircuitState.OPEN and
            time.monotonic() - self._last_fail >= self.recovery_timeout
        )

    async def _on_success(self):
        async with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info(f"circuit_breaker.closed name={self.name}")
            self._failures = 0
            self._state    = CircuitState.CLOSED

    async def _on_failure(self):
        async with self._lock:
            self._failures  += 1
            self._last_fail  = time.monotonic()
            if (self._state == CircuitState.HALF_OPEN or
                    self._failures >= self.failure_threshold):
                self._state = CircuitState.OPEN
                logger.warning(
                    f"circuit_breaker.open name={self.name} failures={self._failures}"
                )

    async def call(self, fn: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        async with self._lock:
            if self._should_recover():
                self._state = CircuitState.HALF_OPEN
                logger.info(f"circuit_breaker.half_open name={self.name}")
            if self._state == CircuitState.OPEN:
                retry_in = self.recovery_timeout - (time.monotonic() - self._last_fail)
                raise ServiceUnavailableError(
                    f"'{self.name}' circuit OPEN — retry in {max(0, retry_in):.0f}s"
                )

        try:
            result = await fn(*args, **kwargs)
            await self._on_success()
            return result
        except ServiceUnavailableError:
            raise
        except Exception:
            await self._on_failure()
            raise

    def protect(self, fn):
        """Decorator to wrap a coroutine with this circuit breaker."""
        @wraps(fn)
        async def wrapper(*a, **k):
            return await self.call(fn, *a, **k)
        return wrapper


# ── Pre-built singletons for each external service ─────────────────────────────
gemini_breaker = CircuitBreaker("gemini", failure_threshold=5,  recovery_timeout=60)
qdrant_breaker  = CircuitBreaker("qdrant", failure_threshold=3,  recovery_timeout=30)
tavily_breaker  = CircuitBreaker("tavily", failure_threshold=5,  recovery_timeout=120)
