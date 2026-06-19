"""
ChatService — thin orchestrator that coordinates all chat dependencies.

Flow:
    1. Save user message → get inserted_id
    2. Load history (cache-aware via HistoryService)
    3. Fetch MCP context (resources + prompts)
    4. Build system prompt via PromptBuilder (with skills listing)
    5. Run Supervisor graph (routes to specialist subgraph)
    6. Stream SSE events to client
    7. Save AI response + token costs
    8. Async memory extraction (non-blocking)
    9. Invalidate history cache
"""

import json
import logging
import asyncio
from datetime import datetime
from bson import ObjectId

from config.model_config import ModelConfig
DEFAULT_MODEL = ModelConfig.DEFAULT_MODEL

from langchain_core.messages import HumanMessage, SystemMessage

from core.database import messages_collection, conversations_collection
from services.history_service import HistoryService
from services.prompt_builder import PromptBuilder
from services.memory_service import MemoryService
from graph.builder import get_agent_graph
from graph.nodes.common import ChatState
from utils.mcp_connection_manager import mcp_manager

import structlog
logger = structlog.get_logger(__name__)


class ChatService:

    @staticmethod
    async def _ensure_conversation(
        conversation_id: str | None,
        user_id: str,
        message: str,
        mcp_server_urls: list[str] | None
    ) -> str:
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
        for url in (urls or []):
            await mcp_manager.connect(url)

    @staticmethod
    async def _fetch_mcp_context() -> tuple[list[dict], list[dict]]:
        try:
            resources = await mcp_manager.get_available_resources()
            prompts   = await mcp_manager.get_available_prompts()
            return resources, prompts
        except Exception as e:
            logger.warning(f"Failed to fetch MCP context: {e}")
            return [], []

    @staticmethod
    async def _bg_upload_to_cloudinary(file_path_str: str, filename: str, user_id: str):
        try:
            from utils.cloudinary_handler import CloudinaryHandler
            from core.database import user_outputs_collection
            from datetime import datetime
            handler = CloudinaryHandler()
            url, public_id = await handler.upload_file(file_path_str, folder=f"chatbot/outputs/{user_id}")
            await user_outputs_collection.update_one(
                {"user_id": user_id, "filename": filename},
                {"$set": {
                    "cloudinary_url": url,
                    "public_id": public_id,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {e}")

    @classmethod
    async def stream(
        cls,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        mcp_server_urls: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        enabled_tools: list[str] | None = None,
        selected_files: list[str] | None = None,
        files_content_parts: list[dict] | None = None,
        attachments: list[dict] | None = None,
    ):
        """
        Main streaming generator. Yields SSE-formatted strings.
        Caller wraps this in a StreamingResponse with media_type="text/event-stream".
        """
        enabled_tools       = enabled_tools or []
        files_content_parts = files_content_parts or []

        try:
            # ── Step 1: Conversation ────────────────────────────────────
            conversation_id = await cls._ensure_conversation(
                conversation_id, user_id, message, mcp_server_urls
            )

            # ── Step 2: Save user message ───────────────────────────────
            result = await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id":         user_id,
                "role":            "user",
                "content":         message,
                "attachments":     attachments or None,
                "timestamp":       datetime.now()
            })
            inserted_user_msg_id = result.inserted_id

            # ── Step 3: Connect MCP servers ─────────────────────────────
            await cls._connect_mcp_servers(mcp_server_urls)

            # ── Step 4: Load history (Redis-cached) ─────────────────────
            history = await HistoryService.get_history(
                conversation_id, user_id,
                exclude_msg_id=inserted_user_msg_id
            )

            # Semantic memory — only retrieve relevant memories
            user_memories = await MemoryService.get_relevant_memories(user_id, message)

            # ── Step 5: Build system prompt ─────────────────────────────
            mcp_resources, mcp_prompts = await cls._fetch_mcp_context()
            system_prompt = PromptBuilder.assemble(
                enabled_tools=enabled_tools,
                mcp_resources=mcp_resources,
                mcp_prompts=mcp_prompts,
                user_memories=user_memories,
                # Note: active_skill_body is injected by the subgraph directly
            )

            # ── Step 6: Build supervisor input ──────────────────────────
            current_content = [{"type": "text", "text": message}] + files_content_parts
            if attachments:
                file_list = ", ".join(f"uploads/{a.get('original_name', 'file')}" for a in attachments)
                note = f"\n\n[Uploaded file(s) available in your sandbox at: {file_list}]"
                current_content[0]["text"] += note
                message += note

            input_message   = HumanMessage(
                content=current_content if files_content_parts else message
            )

            agent_input: ChatState = {
                "messages":        [SystemMessage(content=system_prompt)] + history + [input_message],
                "user_id":         user_id,
                "conversation_id": conversation_id,
                "enabled_tools":   enabled_tools,
                "selected_files":  selected_files,
            }

            config = {
                "run_name": f"supervisor | user={user_id[:8]} | conv={conversation_id[:8]}",
                "tags":     [f"user:{user_id}", f"conv:{conversation_id}", f"model:{model}"],
                "metadata": {
                    "user_id":         user_id,
                    "conversation_id": conversation_id,
                    "model":           model,
                    "enabled_tools":   enabled_tools,
                    "has_files":       bool(files_content_parts),
                },
                "configurable": {
                    "thread_id":     conversation_id,   # Redis checkpointer key
                    "enabled_tools": enabled_tools,
                    "user_id":       user_id,
                    "model":         model,
                },
                "recursion_limit": 30,
            }

            # ── Step 7: Stream supervisor events ────────────────────────
            full_response       = ""
            tool_steps          = []
            skills              = []
            artifacts           = []
            total_input_tokens  = 0
            total_output_tokens = 0
            routed_agent        = "agentx"

            # Gemini 2.5 Flash pricing (USD per token)
            INPUT_PRICE_PER_TOKEN  = 0.075 / 1_000_000
            OUTPUT_PRICE_PER_TOKEN = 0.30  / 1_000_000

            agent_graph = await get_agent_graph()

            # Snapshot existing files BEFORE agent runs — name -> mtime, so we catch
            # both brand-new files AND overwritten files (same name, updated content)
            from utils.workspace import workspace_for as _ws_for
            _CREATED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html", ".png", ".jpg", ".svg", ".md", ".json"}
            _outputs_dir = _ws_for(user_id) / "outputs"
            _outputs_dir.mkdir(parents=True, exist_ok=True)
            _files_before = {
                f.name: f.stat().st_mtime
                for f in _outputs_dir.iterdir()
                if f.is_file() and f.suffix.lower() in _CREATED_EXT
            }

            async for event in agent_graph.astream_events(agent_input, version="v2", config=config):
                if not isinstance(event, dict):
                    continue

                event_type = event.get("event")
                node_name  = event.get("metadata", {}).get("langgraph_node", "")

                # Stream text chunks from any subgraph's chat model
                if event_type == "on_chat_model_stream" and node_name == "agent_node":
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
                        if text:
                            full_response += text
                            yield f"data: {json.dumps({'chunk': text})}\n\n"

                # Tool usage events
                elif event_type == "on_tool_start":
                    tool_name = event.get("name")
                    tool_args = event.get("data", {}).get("input")
                    yield f"data: {json.dumps({'status': f'Using tool: {tool_name}'})}\n\n"
                    yield f"data: {json.dumps({'tool_call': {'name': tool_name, 'args': tool_args}})}\n\n"
                    tool_steps.append({"name": tool_name, "args": tool_args, "status": "running"})

                    if tool_name == "load_skill":
                        skill_name = (tool_args or {}).get("skill_name", "")
                        if skill_name:
                            # Load the actual SKILL.md content so the frontend can display it
                            try:
                                from skills.skill_loader import load_builtin_skill
                                skill_content = load_builtin_skill(skill_name) or ''
                            except Exception:
                                skill_content = ''
                            skill_data = {'name': skill_name, 'content': skill_content}
                            skills.append(skill_data)
                            yield f"data: {json.dumps({'skill_used': skill_data})}\n\n"

                elif event_type == "on_tool_end":
                    tool_name = event.get("name")
                    output    = event.get("data", {}).get("output", "")
                    yield f"data: {json.dumps({'tool_output': {'name': tool_name, 'result': str(output)}})}\n\n"
                    
                    # Intercept artifact creation
                    if tool_name in ["write_to_file", "create_pdf", "create_docx", "create_pptx", "run_python"]:
                        matched_args = {}
                        for step in reversed(tool_steps):
                            if step["name"] == tool_name and step["status"] == "running":
                                step["result"] = str(output)
                                step["status"] = "completed"
                                matched_args = step.get("args", {})
                                break
                        
                        # For run_python, parse file path from the output string
                        if tool_name == "run_python":
                            out_str = str(output)
                            # Look for common file creation patterns in the output
                            import re
                            file_match = re.search(r'(?:saved?|created?|written?|output).*?[:\s]+([\w./\\-]+\.(?:pdf|docx|pptx|xlsx|csv|txt|html|png|jpg|md|json))', out_str, re.IGNORECASE)
                            if file_match:
                                file_path = file_match.group(1).strip()
                                content = matched_args.get('code', '')
                                
                                # Try to read the actual file content if it's text-based
                                if file_path.lower().endswith(('.md', '.txt', '.csv', '.json', '.html')):
                                    try:
                                        from utils.workspace import workspace_for
                                        ws = workspace_for(user_id)
                                        actual_file = ws / file_path
                                        if actual_file.exists():
                                            content = actual_file.read_text(encoding='utf-8')
                                    except Exception:
                                        pass

                                artifact_data = {'name': file_path, 'content': content, 'tool': tool_name}
                                artifacts.append(artifact_data)
                                yield f"data: {json.dumps({'artifact_created': artifact_data})}\n\n"
                        else:
                            file_path = matched_args.get("file_path", "") or matched_args.get("target_file", "") or matched_args.get("output_path", "")
                            file_content = matched_args.get("content", "") or matched_args.get("code", "")
                            if "error" not in str(output).lower() and file_path:
                                artifact_data = {'name': file_path, 'content': file_content, 'tool': tool_name}
                                artifacts.append(artifact_data)
                                yield f"data: {json.dumps({'artifact_created': artifact_data})}\n\n"
                    else:
                        for step in reversed(tool_steps):
                            if step["name"] == tool_name and step["status"] == "running":
                                step["result"] = str(output)
                                step["status"] = "completed"
                                break
                                
                elif event_type == "on_custom_event" and event.get("name") == "exec_output":
                    data = event.get("data", {})
                    yield f"data: {json.dumps({'exec_output': data})}\n\n"


                # Token tracking and final text fallback
                elif event_type == "on_chat_model_end" and node_name == "agent_node":
                    output_msg = event.get("data", {}).get("output")
                    if output_msg:
                        # Fallback: if no text was streamed, capture it here
                        content = getattr(output_msg, "content", "")
                        if content and isinstance(content, str) and content not in full_response:
                            full_response += content
                            yield f"data: {json.dumps({'chunk': content})}\n\n"
                        elif isinstance(content, list):
                            text_parts = [p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content if isinstance(p, dict) and "text" in p or isinstance(p, str)]
                            text_str = "".join(text_parts)
                            if text_str and text_str not in full_response:
                                full_response += text_str
                                yield f"data: {json.dumps({'chunk': text_str})}\n\n"

                        if hasattr(output_msg, "usage_metadata") and output_msg.usage_metadata:
                            usage = output_msg.usage_metadata
                        total_input_tokens  += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                        logger.info(
                            "token.usage",
                            user_id=user_id,
                            conversation_id=conversation_id,
                            node=node_name,
                            model=model,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )

            # ── Step 7b: Detect files created/updated during this agent run ──────
            created_files = []
            try:
                if _outputs_dir.exists():
                    for f in sorted(_outputs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                        if not f.is_file() or f.suffix.lower() not in _CREATED_EXT:
                            continue
                        prev_mtime = _files_before.get(f.name)
                        curr_mtime = f.stat().st_mtime
                        # Show if: file is brand-new OR file was overwritten (mtime changed)
                        if prev_mtime is None or curr_mtime > prev_mtime:
                            created_files.append({
                                "name":         f.name,
                                "size_bytes":   f.stat().st_size,
                                "download_url": f"/outputs/my/{f.name}",
                                "ext":          f.suffix.lower().lstrip("."),
                            })
                            # Launch background upload to Cloudinary for persistence
                            import asyncio
                            asyncio.create_task(cls._bg_upload_to_cloudinary(str(f), f.name, user_id))
                if created_files:
                    yield f"data: {json.dumps({'files_created': created_files})}\n\n"
            except Exception as _fe:
                logger.warning(f"File detection failed (non-fatal): {_fe}")

            # ── Step 8: Save AI response ────────────────────────────────
            cost_usd = (
                total_input_tokens  * INPUT_PRICE_PER_TOKEN +
                total_output_tokens * OUTPUT_PRICE_PER_TOKEN
            )
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id":         user_id,
                "role":            "model",
                "content":         full_response,
                "tool_steps":      tool_steps,
                "skills":          skills,
                "artifacts":       artifacts,
                "files_created":   created_files,
                "model":           model,
                "input_tokens":    total_input_tokens,
                "output_tokens":   total_output_tokens,
                "cost_usd":        round(cost_usd, 8),
                "timestamp":       datetime.now()
            })

            # ── Step 9: Async memory extraction ─────────────────────────
            import asyncio
            asyncio.create_task(
                MemoryService.extract_and_store(
                    user_id=user_id,
                    human_message=message,
                    ai_response=full_response,
                )
            )

            await HistoryService.invalidate(conversation_id)
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'agent': routed_agent})}\n\n"

        except Exception as e:
            logger.error(f"ChatService.stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
