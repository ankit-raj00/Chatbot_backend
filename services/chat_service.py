"""
ChatService — thin orchestrator that coordinates all chat dependencies.
Replaces the 300-line process_chat_stream function in chat_controller.py.

Flow:
    1. Save user message → get inserted_id
    2. Load history (cache-aware via HistoryService)
    3. Fetch MCP context (resources + prompts)
    4. Build system prompt via PromptBuilder
    5. Run LangGraph graph
    6. Stream SSE events to client
    7. Save AI response + invalidate history cache
"""

import json
import logging
from datetime import datetime
from bson import ObjectId

from langchain_core.messages import HumanMessage, SystemMessage

from core.database import messages_collection, conversations_collection
from services.history_service import HistoryService
from services.prompt_builder import PromptBuilder
from services.memory_service import MemoryService
from graph.builder import chat_graph
from utils.mcp_connection_manager import mcp_manager

logger = logging.getLogger(__name__)


class ChatService:

    @staticmethod
    async def _ensure_conversation(
        conversation_id: str | None,
        user_id: str,
        message: str,
        mcp_server_urls: list[str] | None
    ) -> str:
        """Return existing conversation_id or create a new conversation. Returns str ID."""
        if conversation_id:
            await conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"updated_at": datetime.now()}}
            )
            return conversation_id

        result = await conversations_collection.insert_one({
            "user_id": user_id,
            "title": message[:50],
            "mcp_server_url": mcp_server_urls[0] if mcp_server_urls else None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        return str(result.inserted_id)

    @staticmethod
    async def _connect_mcp_servers(urls: list[str]) -> None:
        """Ensure all requested MCP servers are connected."""
        for url in (urls or []):
            await mcp_manager.connect(url)

    @staticmethod
    async def _fetch_mcp_context() -> tuple[list[dict], list[dict]]:
        """Fetch MCP resources and prompts. Returns (resources, prompts)."""
        try:
            resources = await mcp_manager.get_available_resources()
            prompts = await mcp_manager.get_available_prompts()
            return resources, prompts
        except Exception as e:
            logger.warning(f"Failed to fetch MCP context: {e}")
            return [], []

    @classmethod
    async def stream(
        cls,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        mcp_server_urls: list[str] | None = None,
        model: str = "gemini-2.5-flash",
        enabled_tools: list[str] | None = None,
        selected_files: list[str] | None = None,
        files_content_parts: list[dict] | None = None,
        attachments: list[dict] | None = None,
    ):
        """
        Main streaming generator. Yields SSE-formatted strings.
        Caller wraps this in a StreamingResponse with media_type="text/event-stream".
        """
        enabled_tools = enabled_tools or []
        files_content_parts = files_content_parts or []

        try:
            # ── Step 1: Conversation ────────────────────────────────────
            conversation_id = await cls._ensure_conversation(
                conversation_id, user_id, message, mcp_server_urls
            )

            # ── Step 2: Save user message ───────────────────────────────
            result = await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "attachments": attachments or None,
                "timestamp": datetime.now()
            })
            inserted_user_msg_id = result.inserted_id

            # ── Step 3: Connect MCP servers ─────────────────────────────
            await cls._connect_mcp_servers(mcp_server_urls)

            # ── Step 4: Load history (Redis-cached) ─────────────────────
            history = await HistoryService.get_history(
                conversation_id, user_id,
                exclude_msg_id=inserted_user_msg_id
            )

            # Fetch user memories for system prompt injection
            user_memories = await MemoryService.get_user_memories(user_id)

            # ── Step 5: Build system prompt ─────────────────────────────
            mcp_resources, mcp_prompts = await cls._fetch_mcp_context()
            system_prompt = PromptBuilder.assemble(
                enabled_tools=enabled_tools,
                mcp_resources=mcp_resources,
                mcp_prompts=mcp_prompts,
                user_memories=user_memories,   # Phase 8 will populate this
            )

            # ── Step 6: Build graph input ───────────────────────────────
            current_content = [{"type": "text", "text": message}] + files_content_parts
            input_message = HumanMessage(content=current_content if files_content_parts else message)

            graph_input = {
                "messages": [SystemMessage(content=system_prompt)] + history + [input_message],
                "selected_files": selected_files,
            }

            config = {
                "configurable": {
                    "enabled_tools": enabled_tools,
                    "user_id": user_id,
                    "model": model,
                }
            }

            # ── Step 7: Stream graph events ─────────────────────────────
            full_response = ""
            tool_steps = []

            async for event in chat_graph.astream_events(graph_input, version="v1", config=config):
                if not isinstance(event, dict):
                    continue

                event_type = event.get("event")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        text = ""
                        if isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, str):
                                    text += part
                                elif isinstance(part, dict) and "text" in part:
                                    text += part["text"]
                        else:
                            text = str(chunk.content)
                        full_response += text
                        yield f"data: {json.dumps({'chunk': text})}\n\n"

                elif event_type == "on_tool_start":
                    tool_name = event.get("name")
                    tool_args = event.get("data", {}).get("input")
                    yield f"data: {json.dumps({'status': f'Using tool: {tool_name}'})}\n\n"
                    yield f"data: {json.dumps({'tool_call': {'name': tool_name, 'args': tool_args}})}\n\n"
                    tool_steps.append({"name": tool_name, "args": tool_args, "status": "running"})

                elif event_type == "on_tool_end":
                    tool_name = event.get("name")
                    output = event.get("data", {}).get("output", "")
                    yield f"data: {json.dumps({'tool_output': {'name': tool_name, 'result': str(output)}})}\n\n"
                    for step in reversed(tool_steps):
                        if step["name"] == tool_name and step["status"] == "running":
                            step["result"] = str(output)
                            step["status"] = "completed"
                            break

            # ── Step 8: Save AI response + invalidate cache ─────────────
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "model",
                "content": full_response,
                "tool_steps": tool_steps,
                "timestamp": datetime.now()
            })

            # Trigger async memory extraction (non-blocking, errors are caught internally)
            import asyncio
            asyncio.create_task(
                MemoryService.extract_and_store(
                    user_id=user_id,
                    human_message=message,
                    ai_response=full_response,
                )
            )

            # Invalidate history cache so next turn gets fresh data
            await HistoryService.invalidate(conversation_id)

            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            logger.error(f"ChatService.stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
