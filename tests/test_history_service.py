"""
Tests for services/history_service.py — real dev Mongo Atlas cluster (see
tests/conftest.py), using disposable conversation/user ids cleaned up after.

Regression coverage for a real production bug: get_history's Mongo query
sorted ascending then limited to MAX_HISTORY_MESSAGES, which returns the
OLDEST N messages of a conversation instead of the most recent N — any
conversation past 30 stored messages silently lost all context past roughly
the 15th exchange, for the rest of its life. No test existed to catch this.

All three scenarios live in one test function deliberately: core/database.py's
Motor client is a module-level singleton bound to whichever event loop first
uses it, and pytest-asyncio's default is a fresh loop per test function,
which breaks that client on a second test in the same file ("Event loop is
closed") — this is the first test file to make real Motor calls across
multiple test functions (test_credit_service.py mocks Mongo/Redis entirely).
One function sidesteps the cross-function loop-lifetime issue outright.
"""
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from motor.motor_asyncio import AsyncIOMotorClient

from core.cache import cache_delete
import services.history_service as history_service_module
from services.history_service import HistoryService, MAX_HISTORY_MESSAGES, HISTORY_CACHE_PREFIX

USER_ID = "test_history_service_probe_user"


async def _seed(collection, conv_id: str, n: int):
    base = datetime.now(timezone.utc)
    docs = [{
        "conversation_id": conv_id, "user_id": USER_ID,
        "role": "user" if i % 2 == 0 else "assistant",
        "content": f"MESSAGE_{i}",
        "timestamp": base + timedelta(seconds=i),
    } for i in range(n)]
    await collection.insert_many(docs)


@pytest.mark.asyncio
async def test_history_ordering_and_recency():
    # A fresh Motor client, bound to THIS test's own event loop, rather than
    # core.database's module-level singleton — that client gets bound to
    # whichever event loop touches it FIRST across the entire ~148-test
    # suite, and Motor clients can't survive their original loop closing.
    # HistoryService.get_history still goes through the real shared client
    # internally (that's what's under test); this one is only used to
    # seed/clean up test fixtures without inheriting that fragility.
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
    test_client = AsyncIOMotorClient(mongo_uri)
    messages_collection = test_client.gemini_mcp_chat.messages

    # get_history() itself reads through history_service's OWN bound name for
    # the shared client (`from core.database import messages_collection`),
    # captured once at import time — same fragility as above, so it needs
    # the same fresh-client treatment for the read path under test, not just
    # the seeding above.
    patcher = patch.object(history_service_module, "messages_collection", messages_collection)
    patcher.start()

    conv_over = "test_history_over_cap"
    conv_chrono = "test_history_chronological"
    conv_under = "test_history_under_cap"

    try:
        # ── Core regression case: more stored messages than the cap must
        # return the NEWEST slice, not the oldest. ──────────────────────────
        total = MAX_HISTORY_MESSAGES + 5
        await _seed(messages_collection, conv_over, total)
        history = await HistoryService.get_history(conv_over, USER_ID)
        assert len(history) == MAX_HISTORY_MESSAGES
        contents = [m.content for m in history]
        expected = [f"MESSAGE_{i}" for i in range(total - MAX_HISTORY_MESSAGES, total)]
        assert contents == expected, (
            f"expected the {MAX_HISTORY_MESSAGES} MOST RECENT messages in "
            f"chronological order; got {contents[0]}..{contents[-1]}"
        )

        # ── Even the truncated slice must read oldest-to-newest for the LLM
        # (sorting descending without re-reversing would produce the opposite). ──
        await _seed(messages_collection, conv_chrono, 10)
        history = await HistoryService.get_history(conv_chrono, USER_ID)
        assert [m.content for m in history] == [f"MESSAGE_{i}" for i in range(10)]

        # ── Under the cap: everything comes back, still in order. ───────────
        await _seed(messages_collection, conv_under, 5)
        history = await HistoryService.get_history(conv_under, USER_ID)
        assert len(history) == 5
        assert [m.content for m in history] == [f"MESSAGE_{i}" for i in range(5)]
    finally:
        patcher.stop()
        for conv_id in (conv_over, conv_chrono, conv_under):
            await messages_collection.delete_many({"conversation_id": conv_id})
            try:
                await cache_delete(f"{HISTORY_CACHE_PREFIX}:{conv_id}")
            except Exception:
                pass
