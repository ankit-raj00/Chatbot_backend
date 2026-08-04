"""
turn_manager — decouples agent execution from any single HTTP request's
lifecycle, so a browser refresh/disconnect doesn't kill an in-progress
generation.

Before this existed, ChatService.stream() was BOTH the thing driving the
LangGraph agent AND the SSE response body for one specific HTTP request — a
client disconnect (tab refresh/close) raised CancelledError/GeneratorExit
straight into the generator, which propagated into astream_events() and
killed the actual agent run. Now, agent execution runs as a detached
asyncio.Task tracked here; the SSE response for any given HTTP request is
just one of potentially several "viewers" attached to that task's live event
feed (an in-memory fan-out queue) plus a replay of everything already
published before it attached. A fresh request (page reload, or a resume
call from a different tab) can attach a NEW viewer to the SAME still-running
turn and pick up exactly where it left off.

Single-process, in-memory design — matches this app's current single
Docker-container deployment (same assumption already made by the in-memory
MemorySaver checkpointer fallback when REDIS_URL is unset). If the server
process itself dies, in-progress turns die with it, same as before this
feature existed; what this fixes is specifically "the CLIENT went away",
not "the SERVER went away".
"""
import asyncio
from datetime import datetime, timezone

import structlog
logger = structlog.get_logger(__name__)

# How long a finished turn's record stays around after completion, so a
# resume/attach request that arrives just after "done" still gets the tail
# (e.g. the final 'done' event) instead of finding nothing.
_FINISHED_TTL_SECONDS = 120


class TurnManager:
    def __init__(self):
        self._turns: dict[str, dict] = {}

    def start(self, turn_id: str, conversation_id: str, user_id: str, task: asyncio.Task) -> None:
        self._turns[turn_id] = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timeline": [],       # ordered list of raw SSE-payload dicts
            "queues": [],         # list[asyncio.Queue] of live subscribers
            "status": "running",  # running | done | error | stopped
            "task": task,
            "started_at": datetime.now(timezone.utc),
        }

    def active_turn_id_for_conversation(self, conversation_id: str, user_id: str) -> str | None:
        for turn_id, t in self._turns.items():
            if (t["conversation_id"] == conversation_id and t["status"] == "running"
                    and t["user_id"] == user_id):
                return turn_id
        return None

    def publish(self, turn_id: str, event: dict) -> None:
        """Synchronous by design: called between awaits inside the execution
        loop, and asyncio.Queue.put_nowait never blocks (queues here are
        unbounded), so this never needs to itself be a coroutine."""
        t = self._turns.get(turn_id)
        if not t:
            return
        t["timeline"].append(event)
        for q in t["queues"]:
            q.put_nowait(event)

    def finish(self, turn_id: str, status: str) -> None:
        t = self._turns.get(turn_id)
        if not t:
            return
        t["status"] = status
        loop = asyncio.get_event_loop()
        loop.call_later(_FINISHED_TTL_SECONDS, self._turns.pop, turn_id, None)

    def attach(self, turn_id: str):
        """Register a new live subscriber. Returns (queue, backlog, status),
        or None if the turn is unknown (never started, wrong id, or already
        garbage-collected well after it finished). `backlog` is everything
        published so far — replay it before reading from `queue` so the
        caller sees a gap-free timeline regardless of when it attached.

        No race with publish(): neither function awaits anything internally,
        and asyncio is single-threaded/cooperative, so each runs atomically
        with respect to the other — a publish() either fully happens before
        or fully after a given attach(), never interleaved with it.
        """
        t = self._turns.get(turn_id)
        if not t:
            return None
        q: asyncio.Queue = asyncio.Queue()
        t["queues"].append(q)
        return q, list(t["timeline"]), t["status"]

    def detach(self, turn_id: str, q: asyncio.Queue) -> None:
        t = self._turns.get(turn_id)
        if t and q in t["queues"]:
            t["queues"].remove(q)

    def request_stop(self, conversation_id: str, user_id: str) -> bool:
        """Cancel the running task for this conversation's active turn, if
        any AND it belongs to this user. Returns True if a turn was found,
        owned by this user, and cancellation was requested (cancellation is
        cooperative — the task stops at its next await point, where
        _run_turn's except-CancelledError handler persists whatever
        streamed so far and publishes a 'stopped' event)."""
        turn_id = self.active_turn_id_for_conversation(conversation_id, user_id)
        if not turn_id:
            return False
        self._turns[turn_id]["task"].cancel()
        return True


turn_manager = TurnManager()
