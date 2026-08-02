"""
Background task registry.

asyncio only keeps a *weak* reference to tasks created with create_task, so a
fire-and-forget `asyncio.create_task(coro)` can be garbage-collected and cancelled
mid-flight under load. `spawn()` retains a strong reference until completion and
surfaces failures via structured logging instead of silently swallowing them.
"""
import asyncio
import structlog

logger = structlog.get_logger(__name__)

# Strong references so the event loop can't GC in-flight tasks.
_TASKS: set[asyncio.Task] = set()


def spawn(coro, name: str = "bg_task") -> asyncio.Task:
    """Schedule a coroutine as a tracked background task.

    Keeps a strong ref until done and logs cancellation/exceptions.
    Use this instead of a bare asyncio.create_task for fire-and-forget work.
    """
    task = asyncio.create_task(coro, name=name)
    _TASKS.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _TASKS.discard(t)
        if t.cancelled():
            logger.warning("bg_task.cancelled", task=name)
            return
        exc = t.exception()
        if exc is not None:
            logger.error("bg_task.failed", task=name, error=str(exc), exc_info=exc)

    task.add_done_callback(_on_done)
    return task
