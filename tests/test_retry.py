"""
Tests for ChatService.retry() — regenerating an assistant response in place.

Everything is one test function for the same event-loop-singleton reason as
test_history_service.py / test_conversation_summary_service.py (see those
files' docstrings): core/database.py's Motor client binds to whichever loop
touches it first across the whole suite.

The turn lock assertions are the point of this file. _precheck_and_lock leaves
the lock HELD on success, and it is normally released only by _run_turn's
finally block — which never runs when retry() returns early. A missed release
locks the user out of chatting until the lock's TTL expires, so every early
return is checked for it explicitly.
"""
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

import services.chat_service as chat_module
import services.history_service as history_module
from services.chat_service import ChatService
from services.credit_service import CreditService
from services.history_service import HistoryService
from services.turn_manager import turn_manager

USER_ID = "test_retry_probe_user"


async def _drain(agen):
    """Collect every SSE frame a retry() call yields."""
    return [chunk async for chunk in agen]


@pytest.mark.asyncio
async def test_retry_edge_cases_and_happy_path():
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
    test_client = AsyncIOMotorClient(mongo_uri)
    messages_collection = test_client.gemini_mcp_chat.messages

    lock_events = []
    run_turn_calls = []

    async def fake_has_credit(user_id):
        return True

    async def fake_acquire(user_id):
        lock_events.append("acquire")
        return True

    async def fake_release(user_id):
        lock_events.append("release")

    async def fake_run_turn(turn_id, *args, **kwargs):
        run_turn_calls.append(args)
        turn_manager.publish(turn_id, {"done": True, "message_id": "stub"})
        turn_manager.finish(turn_id, "done")

    async def fake_invalidate(conversation_id):
        return None

    patchers = [
        patch.object(chat_module, "messages_collection", messages_collection),
        patch.object(history_module, "messages_collection", messages_collection),
        patch.object(CreditService, "has_credit", staticmethod(fake_has_credit)),
        patch.object(CreditService, "acquire_turn_lock", staticmethod(fake_acquire)),
        patch.object(CreditService, "release_turn_lock", staticmethod(fake_release)),
        patch.object(ChatService, "_run_turn", staticmethod(fake_run_turn)),
        patch.object(HistoryService, "invalidate", staticmethod(fake_invalidate)),
    ]
    for p in patchers:
        p.start()

    conv_oid = ObjectId()
    cid = str(conv_oid)
    base = datetime.now(timezone.utc)

    def _lock_balanced():
        return lock_events.count("acquire") == lock_events.count("release")

    try:
        user_msg = await messages_collection.insert_one({
            "conversation_id": cid, "user_id": USER_ID, "role": "user",
            "content": "what is 2+2?", "timestamp": base,
        })
        model_msg = await messages_collection.insert_one({
            "conversation_id": cid, "user_id": USER_ID, "role": "model",
            "content": "five", "model": "test-model-xyz",
            "timestamp": base + timedelta(seconds=1),
        })

        # ── Unknown message id → error, lock released ───────────────────
        lock_events.clear()
        out = await _drain(ChatService.retry(USER_ID, cid, str(ObjectId())))
        assert "message_not_found" in out[0]
        assert _lock_balanced(), f"lock leaked: {lock_events}"

        # ── Another user's message → error, lock released ───────────────
        lock_events.clear()
        out = await _drain(ChatService.retry("someone_else", cid, str(model_msg.inserted_id)))
        assert "message_not_found" in out[0]
        assert _lock_balanced(), f"lock leaked: {lock_events}"

        # ── A user message is not retryable (role-scoped) ───────────────
        lock_events.clear()
        out = await _drain(ChatService.retry(USER_ID, cid, str(user_msg.inserted_id)))
        assert "message_not_found" in out[0]
        assert _lock_balanced(), f"lock leaked: {lock_events}"

        # ── NOT the last message → retry_not_last, lock released ────────
        later = await messages_collection.insert_one({
            "conversation_id": cid, "user_id": USER_ID, "role": "user",
            "content": "follow-up", "timestamp": base + timedelta(seconds=2),
        })
        lock_events.clear()
        out = await _drain(ChatService.retry(USER_ID, cid, str(model_msg.inserted_id)))
        assert "retry_not_last" in out[0], out
        assert _lock_balanced(), f"lock leaked: {lock_events}"
        # the rejected attempt must NOT have tombstoned anything
        assert (await messages_collection.find_one({"_id": model_msg.inserted_id})).get("superseded") is None
        await messages_collection.delete_one({"_id": later.inserted_id})

        # ── Orphan response (no preceding user message) → lock released ─
        orphan_conv = str(ObjectId())
        orphan = await messages_collection.insert_one({
            "conversation_id": orphan_conv, "user_id": USER_ID, "role": "model",
            "content": "orphaned", "timestamp": base,
        })
        lock_events.clear()
        out = await _drain(ChatService.retry(USER_ID, orphan_conv, str(orphan.inserted_id)))
        assert "prompt_not_found" in out[0], out
        assert _lock_balanced(), f"lock leaked: {lock_events}"
        await messages_collection.delete_many({"conversation_id": orphan_conv})

        # ── Happy path ──────────────────────────────────────────────────
        lock_events.clear()
        run_turn_calls.clear()
        out = await _drain(ChatService.retry(USER_ID, cid, str(model_msg.inserted_id)))
        assert any("done" in c for c in out), out

        # old response tombstoned, NOT deleted (survives a failed regen)
        old = await messages_collection.find_one({"_id": model_msg.inserted_id})
        assert old is not None, "retry hard-deleted the response instead of tombstoning it"
        assert old["superseded"] is True
        assert old.get("superseded_at") is not None

        # _run_turn got the PRECEDING USER message, not the assistant one,
        # and inherited the replaced response's model
        assert len(run_turn_calls) == 1
        args = run_turn_calls[0]
        assert args[1] == "what is 2+2?", f"wrong prompt replayed: {args[1]!r}"
        assert args[4] == "test-model-xyz", f"model not inherited: {args[4]!r}"
        assert args[9] == user_msg.inserted_id, "wrong exclude_msg_id"

        # ── The tombstone is invisible to the agent's context ───────────
        hist = await HistoryService.get_history(cid, USER_ID)
        assert not any("five" in str(m.content) for m in hist), \
            "superseded response leaked back into agent history"

        # ── Retrying an already-superseded message is refused ───────────
        lock_events.clear()
        out = await _drain(ChatService.retry(USER_ID, cid, str(model_msg.inserted_id)))
        assert "message_not_found" in out[0], out
        assert _lock_balanced(), f"lock leaked: {lock_events}"

    finally:
        await messages_collection.delete_many({"conversation_id": cid})
        for p in patchers:
            p.stop()
        test_client.close()
