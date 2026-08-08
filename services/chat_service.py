"""
ChatService — thin orchestrator that coordinates all chat dependencies.

stream() does the fast synchronous setup (resolve conversation, save the
user message) and then hands actual agent execution to _run_turn(), which
runs as a DETACHED background task tracked by services.turn_manager rather
than being tied to the HTTP request that started it. stream() (and resume(),
for a later/different request reattaching to the same turn) just become
"viewers" of that task's live event feed via _view_turn() — so a client
refresh/disconnect only drops one viewer, it doesn't cancel the generation.

_run_turn() flow:
    1. (done in stream(), before spawning) Save user message → get inserted_id
    2. Load history (cache-aware via HistoryService)
    3. Fetch MCP context (resources + prompts)
    4. Build system prompt via PromptBuilder (with skills listing)
    5. Run the agent graph, publishing each event to turn_manager
    6. Save AI response + token costs
    7. Async memory extraction (non-blocking)
    8. Invalidate history cache
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime
from bson import ObjectId

from config.model_config import ModelConfig
DEFAULT_MODEL = ModelConfig.DEFAULT_MODEL

from langchain_core.messages import HumanMessage, SystemMessage

from core.database import messages_collection, conversations_collection
from services.history_service import HistoryService
from services.prompt_builder import PromptBuilder
from services.memory_service import MemoryService
from services.credit_service import CreditService, CREDIT_GRACE_USD
from services.langsmith_service import LangSmithService
from services.turn_manager import turn_manager
from graph.builder import get_agent_graph
from graph.nodes.common import ChatState
from utils.mcp_connection_manager import mcp_manager
from utils.background_tasks import spawn

import structlog
logger = structlog.get_logger(__name__)


class ChatService:

    @staticmethod
    async def _ensure_conversation(
        conversation_id: str | None,
        user_id: str,
        message: str,
        mcp_server_ids: list[str] | None
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
            "mcp_server_id": mcp_server_ids[0] if mcp_server_ids else None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        return str(result.inserted_id)

    @staticmethod
    async def _fetch_mcp_context(user_id: str, mcp_server_ids: list[str]) -> tuple[list[dict], list[dict]]:
        try:
            resources = await mcp_manager.get_available_resources(user_id, mcp_server_ids)
            prompts   = await mcp_manager.get_available_prompts(user_id, mcp_server_ids)
            return resources, prompts
        except Exception as e:
            logger.warning(f"Failed to fetch MCP context: {e}")
            return [], []

    @staticmethod
    async def _bg_upload_to_cloudinary(file_path_str: str, filename: str, user_id: str, conversation_id: str):
        """Persist a generated file to Cloudinary (the durable source of truth) and
        index it in MongoDB. Retries a few times before giving up, because this is
        what makes an output recoverable after a local cache eviction / restart.

        Keyed by (user_id, conversation_id, filename) — NOT just (user_id,
        filename) — so two conversations that happen to generate a
        same-named file (e.g. two unrelated "report.pdf") don't upsert over
        each other's durable Cloudinary record.
        """
        from utils.cloudinary_handler import CloudinaryHandler
        from core.database import user_outputs_collection
        from datetime import datetime, timezone

        handler = CloudinaryHandler()
        last_err = None
        for attempt in range(1, 4):
            try:
                url, public_id = await handler.upload_file(
                    file_path_str, folder=f"chatbot/outputs/{user_id}/{conversation_id}"
                )
                await user_outputs_collection.update_one(
                    {"user_id": user_id, "conversation_id": conversation_id, "filename": filename},
                    {"$set": {
                        "cloudinary_url": url,
                        "public_id": public_id,
                        "updated_at": datetime.now(timezone.utc),
                    }},
                    upsert=True,
                )
                logger.info("output.persisted", filename=filename, user_id=user_id, conversation_id=conversation_id, attempt=attempt)
                return
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.5 * attempt)
        # All retries failed — this output now exists ONLY on local disk (at risk).
        logger.error(
            "output.persist_failed", filename=filename, user_id=user_id, conversation_id=conversation_id, error=str(last_err)
        )

    @staticmethod
    async def _persist_stopped_message(
        cid: str, user_id: str, content: str, timeline: list,
        input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
    ) -> None:
        """Persist a partial assistant message when a turn was stopped/cancelled,
        so the streamed-so-far content isn't lost on reload. A stopped turn
        (whether user-initiated or the credit grace-buffer cutoff) still
        consumed real tokens, so this also records/deducts whatever cost was
        actually accrued — not just $0, or the grace buffer never actually
        gets billed."""
        try:
            await messages_collection.insert_one({
                "conversation_id": cid,
                "user_id":         user_id,
                "role":            "model",
                "content":         content,
                "timeline":        timeline,
                "stopped":         True,
                "input_tokens":    input_tokens,
                "output_tokens":   output_tokens,
                "cost_usd":        round(cost_usd, 8),
                "timestamp":       datetime.now(),
            })
            await HistoryService.invalidate(cid)
        except Exception as e:
            logger.warning(f"failed to persist stopped message: {e}")
        if cost_usd > 0:
            spawn(CreditService.record_and_deduct(user_id, cost_usd), name="credit_deduction_stopped")

    @classmethod
    async def stream(
        cls,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        mcp_server_ids: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        enabled_tools: list[str] | None = None,
        selected_files: list[str] | None = None,
        files_content_parts: list[dict] | None = None,
        attachments: list[dict] | None = None,
    ):
        """
        Entry point for a NEW turn. Does the fast synchronous setup (resolve
        conversation, save the user message), then hands off the actual
        agent execution to a DETACHED background task (see turn_manager) and
        streams it back to the caller. Because execution isn't tied to this
        generator's lifetime, a client disconnect (tab refresh/close) here
        only detaches this one viewer — the agent keeps running, and
        `resume()` can reattach to it later (from this tab reloading or a
        different one) and pick up exactly where it left off.

        Caller wraps this in a StreamingResponse with media_type="text/event-stream".
        """
        enabled_tools       = enabled_tools or []
        mcp_server_ids      = mcp_server_ids or []
        files_content_parts = files_content_parts or []

        # ── Credit pre-check ────────────────────────────────────────────
        # Hard block BEFORE any conversation/message gets created — a blocked
        # turn should leave no trace. Same SSE error channel used everywhere
        # else in this method (see the Step 1-2 except block below), so the
        # frontend's existing `data.error` handling picks this up unchanged;
        # error_code lets it show a specific message instead of the generic
        # fallback (see ChatPage.jsx).
        if not await CreditService.has_credit(user_id):
            cap = await CreditService.get_cap(user_id)
            block_message = f"You've used your ${cap:.0f} free credit. Upgrade to keep chatting."
            yield f"data: {json.dumps({'error': block_message, 'error_code': 'credit_limit_reached'})}\n\n"
            return

        # Steps 1-2 aren't inside _run_turn's try/except (that only guards
        # the spawned background task) — guard them here so a failure this
        # early (bad conversation_id, a Mongo hiccup) still surfaces as a
        # graceful SSE error instead of an unhandled 500 before any bytes
        # are sent.
        try:
            # ── Step 1: Conversation ────────────────────────────────────
            conversation_id = await cls._ensure_conversation(
                conversation_id, user_id, message, mcp_server_ids
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
        except Exception as e:
            logger.error(f"ChatService.stream setup error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        turn_id = str(uuid.uuid4())
        task = spawn(
            cls._run_turn(
                turn_id, user_id, message, conversation_id, mcp_server_ids, model,
                enabled_tools, selected_files, files_content_parts, attachments,
                inserted_user_msg_id,
            ),
            name="agent_turn",
        )
        turn_manager.start(turn_id, conversation_id, user_id, task)

        async for chunk in cls._view_turn(turn_id):
            yield chunk

    @classmethod
    async def resume(cls, conversation_id: str, user_id: str):
        """Reattach to an already-running turn for this conversation (e.g.
        the page was reloaded mid-generation). Yields nothing but a single
        'no active turn' error event if there isn't one — callers should
        check has_active_turn() first to avoid this in the common case, but
        it's handled gracefully either way (e.g. a race where the turn
        finished between the check and this call)."""
        turn_id = turn_manager.active_turn_id_for_conversation(conversation_id, user_id)
        if not turn_id:
            yield f"data: {json.dumps({'error': 'No active generation to resume.'})}\n\n"
            return
        async for chunk in cls._view_turn(turn_id):
            yield chunk

    @staticmethod
    def has_active_turn(conversation_id: str, user_id: str) -> bool:
        return turn_manager.active_turn_id_for_conversation(conversation_id, user_id) is not None

    @staticmethod
    async def stop_turn(conversation_id: str, user_id: str) -> bool:
        return turn_manager.request_stop(conversation_id, user_id)

    @classmethod
    async def _view_turn(cls, turn_id: str):
        """Shared by stream() (the turn's first viewer) and resume() (any
        later viewer, possibly a different tab): attach to the turn's live
        event feed, replay everything already published (so a late attacher
        never sees a gap), then forward new events as they arrive until the
        turn reaches a terminal state. Detaching (on break/return/exception)
        never affects the underlying turn — only removes this one viewer."""
        attached = turn_manager.attach(turn_id)
        if not attached:
            yield f"data: {json.dumps({'error': 'Turn not found or already finished.'})}\n\n"
            return

        queue, backlog, status = attached
        try:
            for event in backlog:
                yield f"data: {json.dumps(event)}\n\n"
            if status != "running":
                # Turn already reached a terminal state before we attached —
                # the backlog above already includes its final event.
                return
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                if "done" in event or "error" in event or "stopped" in event:
                    break
        finally:
            turn_manager.detach(turn_id, queue)

    @classmethod
    async def _run_turn(
        cls,
        turn_id: str,
        user_id: str,
        message: str,
        conversation_id: str,
        mcp_server_ids: list[str],
        model: str,
        enabled_tools: list[str],
        selected_files: list[str] | None,
        files_content_parts: list[dict],
        attachments: list[dict] | None,
        inserted_user_msg_id,
    ) -> None:
        """The actual agent execution — runs as a detached background task
        (see stream()), publishing every event to turn_manager instead of
        yielding directly. Any number of viewers (stream()'s original caller,
        resume() reattaching after a refresh, even a second tab) can watch
        this run live or catch up on it; none of them can cancel it just by
        disconnecting — only stop_turn() can, which cancels THIS task."""
        try:
            # ── Step 4: Load history (Redis-cached) ─────────────────────
            history = await HistoryService.get_history(
                conversation_id, user_id,
                exclude_msg_id=inserted_user_msg_id
            )

            # Semantic memory — only retrieve relevant memories
            user_memories = await MemoryService.get_relevant_memories(user_id, message)

            # ── Step 5: Build system prompt ─────────────────────────────
            mcp_resources, mcp_prompts = await cls._fetch_mcp_context(user_id, mcp_server_ids)
            system_prompt = PromptBuilder.assemble(
                enabled_tools=enabled_tools,
                mcp_resources=mcp_resources,
                mcp_prompts=mcp_prompts,
                user_memories=user_memories,
                # Note: active_skill_body is injected by the subgraph directly
            )

            # ── Step 6: Build supervisor input ──────────────────────────
            # Uploaded files live in the sandbox. Images are looked at with the
            # analyze_image vision tool (by sandbox path); other files are read
            # with run_python/run_shell.
            current_content = [{"type": "text", "text": message}] + files_content_parts
            if attachments:
                image_atts = [a for a in attachments if a.get("is_image")]
                file_atts  = [a for a in attachments if not a.get("is_image")]
                notes = []
                if image_atts:
                    paths = ", ".join(a.get("sandbox_path", f"uploads/{a.get('original_name', 'image')}") for a in image_atts)
                    notes.append(f"[Image(s) uploaded to your sandbox at: {paths}. "
                                 f"To see them, call analyze_image(sandbox_path, query).]")
                if file_atts:
                    paths = ", ".join(a.get("sandbox_path", f"uploads/{a.get('original_name', 'file')}") for a in file_atts)
                    notes.append(f"[File(s) uploaded to your sandbox at: {paths}. "
                                 f"Read them with run_python/run_shell.]")
                if notes:
                    note = "\n\n" + "\n".join(notes)
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
                "mcp_server_ids":  mcp_server_ids,
                "selected_files":  selected_files,
            }

            config = {
                # Reuses the same turn_id already minted for turn_manager /
                # configurable.turn_id below — this becomes the LangSmith root
                # run's ID, so a real per-turn cost can be attached to it
                # after the turn completes (see LangSmithService.attach_cost).
                "run_id":   uuid.UUID(turn_id),
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
                    "thread_id":      conversation_id,
                    "enabled_tools":  enabled_tools,
                    "mcp_server_ids": mcp_server_ids,
                    "user_id":        user_id,
                    "model":          model,
                    # Unique per TURN (unlike thread_id, which is the stable
                    # conversation_id shared across many turns) — lets
                    # agent_tool_node log an incremental, crash-durable trail
                    # of this specific turn's tool calls as they happen. Same
                    # id turn_manager uses to track this run — one concept,
                    # not two random uuids for the same turn.
                    "turn_id":        turn_id,
                },
                # Each tool-call round trip (agent_node -> agent_tool_node -> agent_node)
                # costs 2 steps. The agent is explicitly instructed to chain many tool
                # calls autonomously for multi-step tasks (see AGENT_SYSTEM_PROMPT), so
                # 30 (~15 tool rounds) was too low for real document/build workflows.
                "recursion_limit": 150,
            }

            # ── Step 7: Stream supervisor events ────────────────────────
            full_response       = ""
            tool_steps          = []
            skills              = []
            artifacts           = []
            total_input_tokens  = 0
            total_output_tokens = 0
            routed_agent        = "agentx"

            # A single chronologically-ordered record of everything that happened
            # this turn (narration text interleaved with tool calls, skills, and
            # artifacts, in the order they actually occurred) — persisted so a
            # reloaded conversation renders identically to the live SSE stream,
            # instead of the lossy "all text at the end, tools grouped separately"
            # shape of full_response/tool_steps/skills/artifacts above.
            timeline: list[dict] = []

            def _tl_add_text(text: str) -> None:
                if timeline and timeline[-1].get("type") == "text":
                    timeline[-1]["content"] += text
                else:
                    timeline.append({"type": "text", "content": text})

            def _tl_complete_tool(tool_name: str, result: str, run_id: str = None) -> None:
                # Match by run_id (unique per tool invocation) when available —
                # matching by name alone picks the LAST "running" entry with
                # that name via reverse scan, which silently mismatches when
                # two calls to the SAME tool are in flight close together
                # (e.g. two analyze_image calls): whichever finishes first
                # gets paired with the wrong (most-recently-added) entry,
                # permanently stranding another entry at status="running"
                # forever — confirmed live (conversation 6a719b5fa11cf10ed1232e28,
                # a trailing analyze_image entry stuck "running" even though the
                # turn itself completed successfully with a 200 response).
                if run_id:
                    for entry in reversed(timeline):
                        if entry.get("type") == "tool" and entry.get("run_id") == run_id:
                            entry["result"] = result
                            entry["status"] = "completed"
                            return
                for entry in reversed(timeline):
                    if entry.get("type") == "tool" and entry.get("name") == tool_name and entry.get("status") == "running":
                        entry["result"] = result
                        entry["status"] = "completed"
                        return

            def _tl_add_exec_output(tool_name: str, line_data: dict, run_id: str = None) -> None:
                if run_id:
                    for entry in reversed(timeline):
                        if entry.get("type") == "tool" and entry.get("run_id") == run_id:
                            entry.setdefault("exec_output", []).append(line_data)
                            return
                for entry in reversed(timeline):
                    if entry.get("type") == "tool" and entry.get("name") == tool_name and entry.get("status") == "running":
                        entry.setdefault("exec_output", []).append(line_data)
                        return

            # Per-token USD pricing from model config (0 when unknown, e.g. via
            # OmniRoute) — avoids reporting wrong costs for arbitrary models.
            _price = ModelConfig.get_pricing(model)
            INPUT_PRICE_PER_TOKEN  = _price["input"]
            OUTPUT_PRICE_PER_TOKEN = _price["output"]

            # Read once, not on every loop iteration below — this turn's mid-
            # generation grace cutoff checks (spend before this turn started)
            # + (cost accrued so far this turn) against (cap + grace buffer).
            # Admins are exempt here too — checked once, same as the pre-turn
            # block in stream(), so a long admin turn can't get killed
            # mid-generation by a cap they're exempt from starting under.
            credit_is_exempt      = await CreditService._is_admin(user_id)
            credit_spend_at_start = 0.0 if credit_is_exempt else await CreditService.get_spend(user_id)
            credit_cap            = 0.0 if credit_is_exempt else await CreditService.get_cap(user_id)

            agent_graph = await get_agent_graph()

            # Snapshot existing files BEFORE agent runs — name -> mtime, so we catch
            # both brand-new files AND overwritten files (same name, updated content).
            # Scoped to THIS conversation's own outputs/ dir (not the whole user) —
            # see utils.workspace.conversation_workspace_for for why: a shared
            # per-user outputs/ dir let one conversation's file write land inside
            # another concurrent conversation's snapshot window and get attached
            # to the wrong reply (confirmed live in prod).
            from utils.workspace import conversation_workspace_for as _conv_ws_for
            _CREATED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html", ".png", ".jpg", ".svg", ".md", ".json"}
            _outputs_dir = _conv_ws_for(user_id, conversation_id) / "outputs"
            _outputs_dir.mkdir(parents=True, exist_ok=True)

            def _snapshot_outputs() -> dict:
                return {
                    f.name: f.stat().st_mtime
                    for f in _outputs_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in _CREATED_EXT
                }

            # sync filesystem scan — off the event loop
            _files_before = await asyncio.to_thread(_snapshot_outputs)

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
                            _tl_add_text(text)
                            turn_manager.publish(turn_id, {'chunk': text})

                # Tool usage events
                elif event_type == "on_tool_start":
                    tool_name = event.get("name")
                    tool_run_id = event.get("run_id")
                    tool_args = event.get("data", {}).get("input")
                    turn_manager.publish(turn_id, {'status': f'Using tool: {tool_name}'})
                    turn_manager.publish(turn_id, {'tool_call': {'name': tool_name, 'args': tool_args}})
                    tool_steps.append({"name": tool_name, "args": tool_args, "status": "running", "run_id": tool_run_id})
                    timeline.append({"type": "tool", "name": tool_name, "args": tool_args, "status": "running", "run_id": tool_run_id})

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
                            timeline.append({"type": "skill", **skill_data})
                            turn_manager.publish(turn_id, {'skill_used': skill_data})

                elif event_type == "on_tool_end":
                    tool_name = event.get("name")
                    tool_run_id = event.get("run_id")
                    output    = event.get("data", {}).get("output", "")
                    _tl_complete_tool(tool_name, str(output), tool_run_id)
                    turn_manager.publish(turn_id, {'tool_output': {'name': tool_name, 'result': str(output)}})

                    # NOTE: a regex-based "artifact" interception used to live here for
                    # write_to_file/create_pdf/create_docx/create_pptx/run_python. Removed:
                    # it guessed a created file's path from the tool's TEXT OUTPUT via
                    # regex, then set the artifact's displayed "content" to the PYTHON
                    # SOURCE CODE that ran (matched_args.get('code', '')) — only reading
                    # the real file back for a handful of text extensions. Binary outputs
                    # (png/pdf/docx/...) got the raw script text as their "content", which
                    # the frontend then tried to render as the file itself — a guaranteed
                    # broken preview for every image an agent ever generated (confirmed
                    # live: conversation 6a71912c70fdcde1c8c17aff, work/transformer_architecture.png
                    # showed a broken image because "content" was the matplotlib script,
                    # not image data). It also fired on work/ scratch files, which the
                    # system prompt explicitly tells the agent are not user-facing —
                    # conflicting with the correct, already-existing mechanism (Step 7's
                    # _detect_created() below) that watches outputs/ specifically and
                    # uploads real files to Cloudinary with a real, renderable URL.
                    # Same run_id-first matching as _tl_complete_tool above — tool_steps
                    # has the identical same-tool-name race risk.
                    def _find_running_step(name: str, run_id: str):
                        if run_id:
                            for s in reversed(tool_steps):
                                if s.get("run_id") == run_id:
                                    return s
                        for s in reversed(tool_steps):
                            if s["name"] == name and s["status"] == "running":
                                return s
                        return None

                    if tool_name == "edit_file":
                        # Parse "Edited <path> (+N -M lines)" so the UI can show a
                        # compact diff pill instead of a wall of code — mirrors the
                        # run_python file-creation regex parsing above.
                        import re as _re
                        diff_match = _re.match(r"Edited (.+) \(\+(\d+) -(\d+) lines\)", str(output))
                        step = _find_running_step(tool_name, tool_run_id)
                        if step:
                            step["result"] = str(output)
                            step["status"] = "completed"
                            if diff_match:
                                step["diff"] = {
                                    "path": diff_match.group(1),
                                    "added": int(diff_match.group(2)),
                                    "removed": int(diff_match.group(3)),
                                }
                        if diff_match:
                            for entry in reversed(timeline):
                                if entry.get("type") == "tool" and entry.get("run_id") == tool_run_id and "diff" not in entry:
                                    entry["diff"] = {
                                        "path": diff_match.group(1),
                                        "added": int(diff_match.group(2)),
                                        "removed": int(diff_match.group(3)),
                                    }
                                    break
                    else:
                        step = _find_running_step(tool_name, tool_run_id)
                        if step:
                            step["result"] = str(output)
                            step["status"] = "completed"

                elif event_type == "on_custom_event" and event.get("name") == "exec_output":
                    data = event.get("data", {})
                    _tl_add_exec_output(data.get("tool", ""), data, event.get("run_id"))
                    turn_manager.publish(turn_id, {'exec_output': data})


                # Token tracking and final text fallback. Fires for BOTH
                # agent_node (the real conversational LLM calls) and
                # agent_tool_node (its periodic _progress_check "stuck loop"
                # judge call, every 10 tool calls) — both are real billable
                # calls on the same model, so both must count toward cost.
                # Only agent_node's output is real chat content though: the
                # judge call's output is a JSON verdict, not something that
                # should ever land in full_response.
                elif event_type == "on_chat_model_end" and node_name in ("agent_node", "agent_tool_node"):
                    output_msg = event.get("data", {}).get("output")
                    if output_msg:
                        if node_name == "agent_node":
                            # Fallback: if no text was streamed, capture it here
                            content = getattr(output_msg, "content", "")
                            if content and isinstance(content, str) and content not in full_response:
                                full_response += content
                                _tl_add_text(content)
                                turn_manager.publish(turn_id, {'chunk': content})
                            elif isinstance(content, list):
                                text_parts = [p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content if isinstance(p, dict) and "text" in p or isinstance(p, str)]
                                text_str = "".join(text_parts)
                                if text_str and text_str not in full_response:
                                    full_response += text_str
                                    _tl_add_text(text_str)
                                    turn_manager.publish(turn_id, {'chunk': text_str})

                        usage = getattr(output_msg, "usage_metadata", None)
                        if usage:
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

                            # Mid-turn grace cutoff: let an already-running
                            # turn spend up to CREDIT_GRACE_USD past the cap
                            # rather than hard-killing it the instant it
                            # crosses (a new turn is blocked outright by the
                            # pre-check above instead). Cancelling here drives
                            # the SAME asyncio.CancelledError path already
                            # used by POST /chat/{id}/stop — partial content
                            # gets persisted, a clean 'stopped' event goes
                            # out, no new protocol needed.
                            turn_cost_so_far = (
                                total_input_tokens  * INPUT_PRICE_PER_TOKEN +
                                total_output_tokens * OUTPUT_PRICE_PER_TOKEN
                            )
                            if not credit_is_exempt and credit_spend_at_start + turn_cost_so_far >= credit_cap + CREDIT_GRACE_USD:
                                logger.warning(
                                    "credit.grace_exceeded — stopping turn",
                                    user_id=user_id, turn_id=turn_id,
                                    spend_at_start=credit_spend_at_start,
                                    turn_cost_so_far=turn_cost_so_far, cap=credit_cap,
                                )
                                asyncio.current_task().cancel()

            # ── Step 7b: Detect files created/updated during this agent run ──────
            created_files = []
            try:
                def _detect_created() -> list[dict]:
                    out = []
                    if not _outputs_dir.exists():
                        return out
                    for f in sorted(_outputs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                        if not f.is_file() or f.suffix.lower() not in _CREATED_EXT:
                            continue
                        prev_mtime = _files_before.get(f.name)
                        curr_mtime = f.stat().st_mtime
                        # Show if: file is brand-new OR file was overwritten (mtime changed)
                        if prev_mtime is None or curr_mtime > prev_mtime:
                            out.append({
                                "name":         f.name,
                                "size_bytes":   f.stat().st_size,
                                "download_url": f"/outputs/my/{conversation_id}/{f.name}",
                                "ext":          f.suffix.lower().lstrip("."),
                                "_path":        str(f),
                            })
                    return out

                # sync filesystem scan — off the event loop
                detected = await asyncio.to_thread(_detect_created)
                for item in detected:
                    # Persist to Cloudinary (durable store) — tracked bg task.
                    spawn(cls._bg_upload_to_cloudinary(item.pop("_path"), item["name"], user_id, conversation_id),
                          name="cloudinary_upload")
                    created_files.append(item)
                if created_files:
                    timeline.append({"type": "files_created", "files": created_files})
                    turn_manager.publish(turn_id, {'files_created': created_files})
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
                "timeline":        timeline,
                "model":           model,
                "input_tokens":    total_input_tokens,
                "output_tokens":   total_output_tokens,
                "cost_usd":        round(cost_usd, 8),
                "timestamp":       datetime.now()
            })

            # ── Step 8b: Deduct credit + attach real cost to the LangSmith
            # run (tracked bg tasks — never block the response on billing or
            # tracing) ───────────────────────────────────────────────────
            if cost_usd > 0:
                spawn(CreditService.record_and_deduct(user_id, cost_usd), name="credit_deduction")
            if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
                spawn(
                    LangSmithService.attach_cost(
                        run_id=turn_id, cost_usd=cost_usd,
                        input_tokens=total_input_tokens, output_tokens=total_output_tokens,
                        model=model,
                    ),
                    name="langsmith_cost_attach",
                )

            # ── Step 9: Async memory extraction (tracked bg task) ───────
            spawn(
                MemoryService.extract_and_store(
                    user_id=user_id,
                    human_message=message,
                    ai_response=full_response,
                ),
                name="memory_extraction",
            )

            await HistoryService.invalidate(conversation_id)
            turn_manager.publish(turn_id, {'done': True, 'conversation_id': conversation_id, 'agent': routed_agent})
            turn_manager.finish(turn_id, "done")

        except asyncio.CancelledError:
            # stop_turn() cancelled us (see turn_manager.request_stop) — this
            # is a NORMAL, expected way for a turn to end now, not a client
            # disconnect: a client disconnecting just detaches a viewer (see
            # _view_turn) and does NOT reach this handler anymore, since this
            # task's lifetime is independent of any one HTTP request. Persist
            # whatever streamed so far, tell any attached viewers it stopped,
            # and swallow the cancellation (a task that catches its own
            # CancelledError and returns completes normally — appropriate
            # here since this is an intentional, successful stop, not a
            # crash).
            partial = locals().get("full_response", "") or ""
            tl = locals().get("timeline", []) or []
            cid = locals().get("conversation_id")
            in_tok  = locals().get("total_input_tokens", 0) or 0
            out_tok = locals().get("total_output_tokens", 0) or 0
            in_price  = locals().get("INPUT_PRICE_PER_TOKEN", 0.0) or 0.0
            out_price = locals().get("OUTPUT_PRICE_PER_TOKEN", 0.0) or 0.0
            partial_cost = in_tok * in_price + out_tok * out_price
            if cid and (partial or tl or in_tok or out_tok):
                await cls._persist_stopped_message(cid, user_id, partial, tl, in_tok, out_tok, partial_cost)
            turn_manager.publish(turn_id, {'stopped': True})
            turn_manager.finish(turn_id, "stopped")

        except Exception as e:
            logger.error(f"ChatService._run_turn error: {e}", exc_info=True)
            raw = str(e)
            low = raw.lower()
            if "recursion" in low and "limit" in low:
                friendly = ("The agent reached its step limit without finishing (it may have gotten "
                            "stuck repeating an action). Try rephrasing your request or asking for a "
                            "simpler approach.")
            elif any(s in low for s in ("connection error", "connecterror", "apiconnectionerror",
                                        "connection refused", "refused", "getaddrinfo", "max retries",
                                        "timed out", "read timeout", "failed to establish")):
                friendly = ("Could not reach the LLM gateway (OmniRoute). Make sure it is running at "
                            "the configured OMNIROUTE_BASE_URL and try again.")
            else:
                friendly = raw

            # Best-effort: persist whatever the assistant produced before failing,
            # so the turn (and the partial timeline) isn't lost entirely.
            try:
                partial = locals().get("full_response", "") or ""
                tl = locals().get("timeline", []) or []
                cid = locals().get("conversation_id")
                if cid and (partial or tl):
                    await messages_collection.insert_one({
                        "conversation_id": cid,
                        "user_id":         user_id,
                        "role":            "model",
                        "content":         partial,
                        "timeline":        tl,
                        "error":           friendly,
                        "timestamp":       datetime.now(),
                    })
                    await HistoryService.invalidate(cid)
            except Exception:
                pass

            turn_manager.publish(turn_id, {'error': friendly})
            turn_manager.finish(turn_id, "error")
