"""Tests for circuit breaker and tool result cache."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock
from utils.circuit_breaker import CircuitBreaker, CircuitState, ServiceUnavailableError


# ── Circuit Breaker Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_circuit_breaker_closed_on_success():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)
    fn = AsyncMock(return_value="ok")
    result = await cb.call(fn)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)
    fn = AsyncMock(side_effect=RuntimeError("API down"))

    for _ in range(3):
        try:
            await cb.call(fn)
        except RuntimeError:
            pass

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_open_raises_service_unavailable():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)
    fn = AsyncMock(side_effect=RuntimeError("fail"))

    try:
        await cb.call(fn)
    except RuntimeError:
        pass

    # Now it's OPEN — next call should fail immediately
    with pytest.raises(ServiceUnavailableError):
        await cb.call(AsyncMock(return_value="ok"))


@pytest.mark.asyncio
async def test_circuit_breaker_protect_decorator():
    cb = CircuitBreaker("test2", failure_threshold=3, recovery_timeout=60)

    @cb.protect
    async def my_fn(x):
        return x * 2

    result = await my_fn(5)
    assert result == 10


# ── Tool Cache Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_cache_non_cacheable_always_executes():
    """Non-cacheable tools must always run the execute_fn."""
    from utils.tool_result_cache import cached_invoke
    call_count = 0

    async def fn():
        nonlocal call_count
        call_count += 1
        return "shell_result"

    # run_shell_command is not in CACHEABLE
    await cached_invoke("run_shell_command", {"cmd": "ls"}, fn)
    await cached_invoke("run_shell_command", {"cmd": "ls"}, fn)
    assert call_count == 2, "Non-cacheable tool should execute every time"
