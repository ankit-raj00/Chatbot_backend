"""
ConversationSummaryService — gives a long conversation continuity past
HistoryService's most-recent-N window instead of hard amnesia.

Context: HistoryService.get_history() loads only the most recent
MAX_HISTORY_MESSAGES (30, see history_service.py) — correctly, after the
2026-08-09 fix that used to return the OLDEST 30 instead. But a 30-message
cap alone still means a long conversation's earlier messages are simply
invisible to the agent once they age out — the fix made the window
CORRECT, not UNLIMITED. This service closes that gap the same way
Anthropic's context-engineering guidance frames it: prefer compaction over
hard truncation, so what falls out of the raw window is remembered as a
summary rather than forgotten outright.

Storage: two fields added to the conversation's OWN document in
conversations_collection (no new collection):
  context_summary: str        — cumulative summary text, updated over time
  summary_covers_count: int   — how many of the conversation's stored
                                 messages have been folded into the summary
                                 so far, so re-runs only summarize what's NEW

Trigger: called as a non-blocking background task after each turn (same
`spawn(..., name=...)` pattern as MemoryService.extract_and_store in
chat_service.py) — never adds latency to the user-visible response, and only
does real (billable) work once a full batch of messages has newly fallen out
of the retained window since the last summarization.

Injection: PromptBuilder.assemble() takes the resulting text as
conversation_summary and includes it as its own section — see
build_conversation_summary_section.
"""
import os
from datetime import datetime

from core.database import conversations_collection, messages_collection

import structlog
logger = structlog.get_logger(__name__)

# HistoryService's own retained-window size — kept in sync by importing it
# rather than duplicating the number.
from services.history_service import MAX_HISTORY_MESSAGES

# Don't summarize one message at a time as soon as it crosses the window
# edge — batch it, so a long conversation pays for one compact LLM call per
# ~10 newly-aged-out messages instead of one call per turn indefinitely.
SUMMARY_BATCH_SIZE = 10

# Hard cap on the summary's own length so IT doesn't become the next thing
# bloating context — old summarized detail should compress further, not
# accumulate forever. Enforced by asking the model to stay under this, not
# by truncating its output (truncating mid-sentence would be worse than the
# amnesia this service exists to fix).
_MAX_SUMMARY_WORDS = 300

SUMMARY_PROMPT = """You maintain a running summary of an ongoing conversation \
between a user and an AI agent, so the agent keeps continuity with parts of \
the conversation that are no longer shown in full.

{existing_block}Here are the NEXT messages in the conversation to fold in:
{new_messages_text}

Write an updated summary that preserves what would matter if the agent needed \
to recall this later: decisions made, facts the user stated about themselves \
or their task, files or results produced, open questions or unfinished work. \
Discard pleasantries, restated context, and anything that's now moot. Keep it \
under {max_words} words. Write PLAIN prose, no headers, no bullet preamble — \
just the summary itself, nothing else."""


class ConversationSummaryService:

    @staticmethod
    async def get_summary(conversation_id: str) -> str:
        """Fetch the current stored summary, or "" if none exists yet."""
        try:
            from bson import ObjectId
            doc = await conversations_collection.find_one(
                {"_id": ObjectId(conversation_id)}, {"context_summary": 1})
            return (doc or {}).get("context_summary", "") or ""
        except Exception as e:
            logger.warning("conversation_summary.get_failed", conversation_id=conversation_id, error=str(e))
            return ""

    @staticmethod
    async def maybe_update_summary(conversation_id: str, user_id: str) -> None:
        """Background task: fold newly-aged-out messages into the running
        summary if a full batch has accumulated since the last update.
        Errors are logged, never raised — this must never break a turn."""
        try:
            from bson import ObjectId

            total = await messages_collection.count_documents({
                "conversation_id": conversation_id, "user_id": user_id,
            })
            # Nothing has fallen out of the retained window yet.
            if total <= MAX_HISTORY_MESSAGES:
                return

            conv_doc = await conversations_collection.find_one(
                {"_id": ObjectId(conversation_id)},
                {"context_summary": 1, "summary_covers_count": 1},
            )
            existing_summary = (conv_doc or {}).get("context_summary", "") or ""
            covered = (conv_doc or {}).get("summary_covers_count", 0) or 0

            # How many messages are newly out-of-window and not yet summarized.
            newly_aged_out = (total - MAX_HISTORY_MESSAGES) - covered
            if newly_aged_out < SUMMARY_BATCH_SIZE:
                return  # Not enough new material yet — skip, save the LLM call.

            cursor = (messages_collection
                      .find({"conversation_id": conversation_id, "user_id": user_id})
                      .sort("timestamp", 1)
                      .skip(covered)
                      .limit(newly_aged_out))
            new_docs = await cursor.to_list(length=newly_aged_out)
            if not new_docs:
                return

            new_messages_text = "\n".join(
                f"{'User' if d.get('role') == 'user' else 'Assistant'}: {d.get('content', '')[:800]}"
                for d in new_docs
            )
            existing_block = (
                f"Existing summary so far:\n{existing_summary}\n\n"
                if existing_summary else ""
            )
            prompt = SUMMARY_PROMPT.format(
                existing_block=existing_block,
                new_messages_text=new_messages_text,
                max_words=_MAX_SUMMARY_WORDS,
            )

            from config.model_config import ModelConfig
            from graph.llm_registry import get_llm
            llm = get_llm(ModelConfig.MEMORY_EXTRACTION_MODEL)
            response = await llm.ainvoke(prompt)
            new_summary = (response.content if isinstance(response.content, str)
                          else str(response.content)).strip()
            if not new_summary:
                return

            await conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {
                    "context_summary": new_summary,
                    "summary_covers_count": covered + len(new_docs),
                    "summary_updated_at": datetime.now(),
                }},
            )
            logger.info("conversation_summary.updated", conversation_id=conversation_id,
                        folded_in=len(new_docs), covers_count=covered + len(new_docs))
        except Exception as e:
            logger.warning("conversation_summary.update_failed",
                            conversation_id=conversation_id, error=str(e), exc_info=True)
