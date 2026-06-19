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

import structlog
logger = structlog.get_logger(__name__)

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
  {{"topic": "tech stack", "content": "Uses Python, FastAPI, and MongoDB"}},
  {{"topic": "project", "content": "Building an AI agent platform called AgentX"}}
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
        if len(human_message) < 5 or len(ai_response) < 5:
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

            logger.info(f"Extracted {len(new_memories)} potential new memories for user {user_id}: {new_memories}")

            # Merge with existing memories
            existing = await MemoryService.get_user_memories(user_id)
            existing_topics = {m.get("topic") for m in existing if m.get("topic")}

            merged = existing[:]
            for mem in new_memories:
                topic = mem.get("topic")
                content = mem.get("content")
                if topic and content:
                    if topic in existing_topics:
                        # Update existing memory for this topic
                        for m in merged:
                            if m.get("topic") == topic:
                                m["content"] = content
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
            logger.info(f"Successfully saved memories to MongoDB for user {user_id}. Now holding {len(merged)} total memories.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"Memory extraction failed for {user_id} (non-fatal): {e}")

    @staticmethod
    async def get_relevant_memories(user_id: str, current_message: str, top_k: int = 5) -> list[dict]:
        """
        Return only the memories most relevant to the current message.
        Uses embedding cosine similarity. Falls back to get_user_memories on error.
        More efficient than injecting all memories — scales to 100s of memories.
        """
        import asyncio
        import numpy as np

        all_mems = await MemoryService.get_user_memories(user_id)
        if len(all_mems) <= top_k:
            return all_mems  # no point filtering small sets

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            import os

            emb = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                output_dimensionality=768,
            )
            texts = [f"{m.get('topic', '')}: {m.get('content', '')}" for m in all_mems]
            q_emb, m_embs = await asyncio.gather(
                emb.aembed_query(current_message[:500]),
                emb.aembed_documents(texts),
            )
            q = np.array(q_emb)
            scored = sorted(
                zip([float(np.dot(q, np.array(me))) for me in m_embs], all_mems),
                reverse=True,
            )
            return [m for _, m in scored[:top_k]]
        except Exception as e:
            logger.warning(f"Semantic memory retrieval failed: {e}")
            return all_mems[:top_k]

    @staticmethod
    async def clear_user_memories(user_id: str) -> None:
        """Delete all memories for a user. Callable from an API endpoint."""
        await user_memories_collection.delete_one({"user_id": user_id})
        logger.info(f"Cleared all memories for user {user_id}")
