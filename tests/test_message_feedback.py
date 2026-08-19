"""
Tests for the thumbs up/down feedback endpoint
(controllers/conversation_controller.py::submit_message_feedback).

Real dev Mongo, disposable ids cleaned up after. Everything lives in ONE test
function for the same reason as test_history_service.py /
test_conversation_summary_service.py: core/database.py's Motor client is a
module-level singleton bound to whichever event loop touches it first across
the whole suite, so independent test functions with fresh event loops break on
it — worked around with a dedicated Motor client patched into the module under
test.
"""
import os
import pytest
from datetime import datetime, timedelta, timezone

from unittest.mock import patch
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from fastapi import HTTPException

import controllers.conversation_controller as cc_module
from controllers.conversation_controller import ConversationController

USER_ID = "test_feedback_probe_user"
OTHER_USER_ID = "test_feedback_probe_other_user"


@pytest.mark.asyncio
async def test_message_feedback_lifecycle_and_ownership():
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
    test_client = AsyncIOMotorClient(mongo_uri)
    messages_collection = test_client.gemini_mcp_chat.messages
    conversations_collection = test_client.gemini_mcp_chat.conversations

    p_msgs = patch.object(cc_module, "messages_collection", messages_collection)
    p_convs = patch.object(cc_module, "conversations_collection", conversations_collection)
    p_msgs.start()
    p_convs.start()

    conv_oid = ObjectId()
    base = datetime.now(timezone.utc)

    try:
        await conversations_collection.insert_one({
            "_id": conv_oid, "user_id": USER_ID, "title": "feedback probe",
            "created_at": base, "updated_at": base,
        })
        user_msg = await messages_collection.insert_one({
            "conversation_id": str(conv_oid), "user_id": USER_ID,
            "role": "user", "content": "hello", "timestamp": base,
        })
        model_msg = await messages_collection.insert_one({
            "conversation_id": str(conv_oid), "user_id": USER_ID,
            "role": "model", "content": "hi there", "timestamp": base + timedelta(seconds=1),
        })

        # ── Thumbs down with a reason persists both fields ──────────────
        await ConversationController.submit_message_feedback(
            str(conv_oid), str(model_msg.inserted_id), USER_ID, "down", "made something up",
        )
        doc = await messages_collection.find_one({"_id": model_msg.inserted_id})
        assert doc["rating"] == "down"
        assert doc["rating_reason"] == "made something up"
        assert doc.get("rated_at") is not None

        # ── Switching to thumbs up overwrites the stale reason ──────────
        # (a leftover "made something up" on an up-rated row would poison
        # the exported dataset with a contradictory label)
        await ConversationController.submit_message_feedback(
            str(conv_oid), str(model_msg.inserted_id), USER_ID, "up", None,
        )
        doc = await messages_collection.find_one({"_id": model_msg.inserted_id})
        assert doc["rating"] == "up"
        assert doc["rating_reason"] is None

        # ── Clearing (toggle-off) sets an explicit None, and that None must
        # be excluded by the export's $in filter ───────────────────────────
        await ConversationController.submit_message_feedback(
            str(conv_oid), str(model_msg.inserted_id), USER_ID, None, None,
        )
        doc = await messages_collection.find_one({"_id": model_msg.inserted_id})
        assert doc["rating"] is None
        assert await messages_collection.count_documents(
            {"_id": model_msg.inserted_id, "rating": {"$in": ["up", "down"]}}
        ) == 0

        # ── A user's OWN prompt is not rateable (role-scoped update) ────
        with pytest.raises(HTTPException) as exc:
            await ConversationController.submit_message_feedback(
                str(conv_oid), str(user_msg.inserted_id), USER_ID, "up", None,
            )
        assert exc.value.status_code == 404

        # ── Another user cannot rate this conversation's messages ───────
        with pytest.raises(HTTPException) as exc:
            await ConversationController.submit_message_feedback(
                str(conv_oid), str(model_msg.inserted_id), OTHER_USER_ID, "down", "not mine",
            )
        assert exc.value.status_code == 404
        # ...and nothing was written by that attempt
        doc = await messages_collection.find_one({"_id": model_msg.inserted_id})
        assert doc["rating"] is None

        # ── Unknown message id in a real conversation → 404, not 500 ────
        with pytest.raises(HTTPException) as exc:
            await ConversationController.submit_message_feedback(
                str(conv_oid), str(ObjectId()), USER_ID, "up", None,
            )
        assert exc.value.status_code == 404

    finally:
        await messages_collection.delete_many({"conversation_id": str(conv_oid)})
        await conversations_collection.delete_one({"_id": conv_oid})
        p_msgs.stop()
        p_convs.stop()
        test_client.close()
