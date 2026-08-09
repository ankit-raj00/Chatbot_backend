"""
Tests for services/conversation_summary_service.py — real dev Mongo Atlas
cluster, real (cheap) LLM call for summarization, disposable conversation ids
cleaned up after. Makes one real LLM call, kept to a single test function for
the same reason as test_history_service.py: core/database.py's Motor client
is a module-level singleton bound to whichever event loop touches it first
across the whole test suite, and breaks across independent test functions'
fresh event loops — worked around with a dedicated Motor client for this
test's own seeding/assertions.
"""
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

import services.conversation_summary_service as summary_module
from services.conversation_summary_service import ConversationSummaryService, SUMMARY_BATCH_SIZE
from services.history_service import MAX_HISTORY_MESSAGES

USER_ID = "test_summary_service_probe_user"


async def _seed(messages_collection, conversations_collection, conv_oid, n: int, fact_at: int = None):
    await conversations_collection.insert_one({
        "_id": conv_oid, "user_id": USER_ID, "title": "probe",
        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
    })
    base = datetime.now(timezone.utc)
    docs = [{
        "conversation_id": str(conv_oid), "user_id": USER_ID,
        "role": "user" if i % 2 == 0 else "assistant",
        "content": f"filler exchange {i}",
        "timestamp": base + timedelta(seconds=i),
    } for i in range(n)]
    if fact_at is not None:
        docs[fact_at]["content"] = "User: my project is called Project Falcon."
    await messages_collection.insert_many(docs)


@pytest.mark.asyncio
async def test_summary_batching_and_content():
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
    test_client = AsyncIOMotorClient(mongo_uri)
    messages_collection = test_client.gemini_mcp_chat.messages
    conversations_collection = test_client.gemini_mcp_chat.conversations

    patcher_msgs = patch.object(summary_module, "messages_collection", messages_collection)
    patcher_convs = patch.object(summary_module, "conversations_collection", conversations_collection)
    patcher_msgs.start()
    patcher_convs.start()

    conv_under = ObjectId()
    conv_over = ObjectId()

    try:
        # ── Below the batch threshold: must NOT trigger a (billable) LLM call. ──
        aged_out_but_below_batch = SUMMARY_BATCH_SIZE - 1
        await _seed(messages_collection, conversations_collection, conv_under,
                    MAX_HISTORY_MESSAGES + aged_out_but_below_batch)
        await ConversationSummaryService.maybe_update_summary(str(conv_under), USER_ID)
        summary = await ConversationSummaryService.get_summary(str(conv_under))
        assert summary == "", "should not summarize below SUMMARY_BATCH_SIZE aged-out messages"

        # ── Over the threshold: must summarize, and the summary must actually
        # capture a fact that lived in an aged-out (no-longer-in-window) message. ──
        fact_index = 3  # well before the retained window once 45 total exist
        await _seed(messages_collection, conversations_collection, conv_over,
                    MAX_HISTORY_MESSAGES + SUMMARY_BATCH_SIZE + 5, fact_at=fact_index)
        before = await ConversationSummaryService.get_summary(str(conv_over))
        assert before == ""

        await ConversationSummaryService.maybe_update_summary(str(conv_over), USER_ID)
        after = await ConversationSummaryService.get_summary(str(conv_over))
        assert after != "", "should have generated a summary once past the batch threshold"
        assert "Falcon" in after, f"summary should mention the aged-out fact, got: {after!r}"

        doc = await conversations_collection.find_one({"_id": conv_over})
        assert doc["summary_covers_count"] == (MAX_HISTORY_MESSAGES + SUMMARY_BATCH_SIZE + 5) - MAX_HISTORY_MESSAGES
    finally:
        for conv_oid in (conv_under, conv_over):
            await messages_collection.delete_many({"conversation_id": str(conv_oid)})
            await conversations_collection.delete_one({"_id": conv_oid})
        patcher_msgs.stop()
        patcher_convs.stop()
