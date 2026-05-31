"""
MemoryService — extracts and persists user-specific facts across conversations.
Inspired by Google ADK Memory Bank pattern.

Storage: MongoDB `user_memories` collection
  Schema: { user_id: str, memories: [{topic, content, created_at}], updated_at }

Extraction: Called asynchronously after AI response is saved.
Injection: Called at start of each chat turn via PromptBuilder.

WHY MongoDB (not Redis): Memories are permanent — they should survive Redis flushes.
WHY LLM extraction: Rule-based extraction misses implicit facts. LLM understands context.
WHY limit to 10 memories: More than 10 memories makes the system prompt too long and noisy.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from core.database import db  # Access the motor client for the new collection

logger = logging.getLogger(__name__)

MAX_MEMORIES = 10   # Maximum memories to retain per user

# Collection reference
user_memories_collection = db["user_memories"]


EXTRACTION_PROMPT = """You are a memory extraction assistant.
Given a conversation snippet, extract 0-3 durable facts about the USER that are worth remembering long-term.
Focus on: name, profession, tech stack, project names, goals, preferences, location.
Do NOT extract: temporary questions, facts about AI, one-off requests.

Return ONLY a JSON array. Each item must have "topic" (1-3 words) and "content" (one sentence).
If there is nothing worth remembering, return an empty array [].
Never include personally sensitive information like passwords or financial details.

Example output:
[
  {"topic": "tech stack", "content": "Uses Python, FastAPI, and MongoDB"},
  {"topic": "project", "content": "Building an AI agent platform called AgentX"}
]

Conversation:
{conversation_text}"""


class MemoryService:

    @staticmethod
    async def get_user_memories(user_id: str) -> list[dict]:
        """Retrieve stored memories for a user. Returns [] if none found."""
        try:
            doc = await user_memories_collection.find_one({"user_id": user_id})
            if doc:
                return doc.get("memories", [])
            return []
        except Exception as e:
            logger.warning(f"Failed to retrieve memories for {user_id}: {e}")
            return []

    @staticmethod
    async def extract_and_store(
        user_id: str,
        human_message: str,
        ai_response: str,
    ) -> None:
        """
        Extract facts from a conversation turn and store in MongoDB.
        Called as a background task — errors are logged, not raised.
        Only runs if the conversation has substantial content (>50 chars each).
        """
        if len(human_message) < 50 or len(ai_response) < 50:
            return   # Too short to have extractable facts

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            import json

            conversation_text = (
                f"User: {human_message[:500]}\n"
                f"Assistant: {ai_response[:500]}"
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",   # Use the cheapest model for this
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )

            prompt = EXTRACTION_PROMPT.format(conversation_text=conversation_text)
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            new_memories = json.loads(raw)
            if not isinstance(new_memories, list):
                return

            if not new_memories:
                return   # Nothing to store

            logger.info(f"Extracted {len(new_memories)} memories for user {user_id}")

            # Merge with existing memories
            existing = await MemoryService.get_user_memories(user_id)
            existing_topics = {m["topic"] for m in existing}

            merged = existing[:]
            for mem in new_memories:
                if mem.get("topic") and mem.get("content"):
                    if mem["topic"] in existing_topics:
                        # Update existing memory for this topic
                        for m in merged:
                            if m["topic"] == mem["topic"]:
                                m["content"] = mem["content"]
                                m["updated_at"] = datetime.now().isoformat()
                    else:
                        mem["created_at"] = datetime.now().isoformat()
                        merged.append(mem)

            # Keep only the most recent MAX_MEMORIES
            merged = merged[-MAX_MEMORIES:]

            # Upsert to MongoDB
            await user_memories_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "user_id": user_id,
                    "memories": merged,
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            logger.info(f"Memory updated for user {user_id}: {len(merged)} total memories")

        except Exception as e:
            logger.warning(f"Memory extraction failed for {user_id} (non-fatal): {e}")

    @staticmethod
    async def clear_user_memories(user_id: str) -> None:
        """Delete all memories for a user. Callable from an API endpoint."""
        await user_memories_collection.delete_one({"user_id": user_id})
        logger.info(f"Cleared all memories for user {user_id}")
