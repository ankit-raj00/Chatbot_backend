"""
VoicePipeline — the core wrapper: audio in, (transcript + response text +
response audio) out. Ties together voice.stt, the EXISTING LangGraph agent
(graph/builder.py — same tools/RAG as chat, not a separate dumbed-down
voice agent), and voice.tts.

Two ways to drive a turn:
  - process_utterance() / process_utterance_streaming(): one-shot REST STT
    on a complete audio blob. Simple, used by the non-live test path.
  - run_utterance_from_stream(): real Deepgram WebSocket streaming STT fed
    by a live audio chunk iterator — the low-latency path routes/voice_routes.py
    actually uses, since it lets the agent start reacting as soon as
    Deepgram finalizes the utterance rather than waiting for a full
    record-then-upload round trip.

Credit gating and conversation/message persistence live here (not in the
route layer) so the route stays a thin WebSocket<->pipeline relay, matching
this codebase's "routes/controllers thin, services do the logic" convention
(see CLAUDE.md).
"""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, List, Optional

import structlog
from bson import ObjectId
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk

from voice import config as voice_config
from voice.stt import transcribe_bytes
from voice.stt import stream_transcribe as _stream_transcribe_deepgram
from voice.sarvam_stt import stream_transcribe as _stream_transcribe_sarvam
from voice.tts import synthesize_stream as _synthesize_stream_cartesia
from voice.sarvam_tts import synthesize_stream as _synthesize_stream_sarvam
from voice.sentence_chunker import chunk_for_tts
from voice.speaker import TurnSpeaker
from graph.builder import get_agent_graph
from graph.nodes.common import ChatState
from config.model_config import ModelConfig
from core.database import messages_collection, conversations_collection
from services.credit_service import CreditService, CREDIT_GRACE_USD

# Same absolute per-turn ceiling services/chat_service.py enforces, applying
# unconditionally even to admin/credit-exempt accounts — a runaway tool loop
# shouldn't be able to spend without limit just because it's a voice turn.
MAX_TURN_COST_USD = float(os.getenv("MAX_TURN_COST_USD", "5.00"))

logger = structlog.get_logger(__name__)

# Resolved once at import time from VOICE_STT_PROVIDER/VOICE_TTS_PROVIDER
# (voice/config.py) — both the Sarvam and Deepgram/Cartesia implementations
# share the exact same async-generator contract (see their respective
# module docstrings), so nothing downstream of these two names needs to
# know or care which provider is actually behind them.
stream_transcribe = (
    _stream_transcribe_sarvam if voice_config.STT_PROVIDER == "sarvam" else _stream_transcribe_deepgram
)
synthesize_stream = (
    _synthesize_stream_sarvam if voice_config.TTS_PROVIDER == "sarvam" else _synthesize_stream_cartesia
)

# Kept identical to services/chat_service.py's set — the two paths write into
# the same outputs/ dir and the same message shape, so a file type visible in
# text chat must be visible in voice chat too.
_CREATED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".html",
                ".png", ".jpg", ".svg", ".md", ".json"}

VOICE_SYSTEM_PROMPT = """You are AgentX, speaking with the user out loud through a voice interface.

Rules for voice responses:
- Speak naturally, like a helpful person talking, not like you're writing a document.
- No markdown, no bullet points, no code blocks, no headings — none of that can be spoken.
- If you use a tool (search their documents, run code, etc.), summarize the result in plain spoken language — never read out raw JSON, file paths, or code.
- "Explain X" / "tell me about X" / "walk me through X" is ALWAYS a request to SPEAK the
  explanation, no matter how long or detailed the topic is — length is not a reason to write a
  file instead of talking. Only write a file (sandbox_run_python writing to outputs/, etc.) when
  the user's own words ask for one: "save this", "make me a PDF/doc/notes", "write this down".
  If you're not sure whether they want a file, ask — don't default to creating one just because
  an explanation is long. A long spoken answer is the correct response to "explain in detail",
  not a shorter spoken answer plus a file with the real content in it.
- Keep genuinely simple answers short. But "explain X in detail" means the detail belongs in
  your SPOKEN answer — don't compress a real explanation down to two sentences and point at a
  file instead just to keep the response brief.

Narrate what you're doing, briefly, as you go — don't go silent while working:
- Before you call a tool, say ONE short sentence about what you're about to do
  ("Let me check your documents for that.", "I'll run that in the sandbox.").
  Say it, THEN call the tool — don't wait until after.
- If a tool call fails or you need to change approach, say so in one short
  line before retrying ("That didn't work, trying a different way.") instead
  of silently retrying.
- Keep every one of these narration lines to a single short sentence — you
  are talking WHILE working, not writing a status report.
"""


@dataclass
class VoiceTurnResult:
    transcript: str
    response_text: str
    audio: bytes
    timing: dict = field(default_factory=dict)


class VoicePipeline:
    """One instance per voice session (one per WebSocket connection). Not
    thread-safe across concurrent utterances by design — a voice session is
    inherently one-at-a-time (you can't process two overlapping utterances
    from the same speaker)."""

    def __init__(self, user_id: str, conversation_id: Optional[str] = None,
                 enabled_tools: Optional[List[str]] = None, mcp_server_ids: Optional[List[str]] = None):
        voice_config.require_keys()
        self.user_id = user_id
        self.conversation_id = conversation_id  # None until first turn creates one
        # Same native-tool/MCP selection the user has toggled on for chat.
        # Previously hardcoded to [] here, which meant voice turns only ever
        # got the always-on sandbox tools + skills — never the user's
        # enabled RAG search, Google Drive, weather, etc., or any MCP
        # server, even though the SAME agent graph as chat fully supports
        # them once given the real lists.
        self.enabled_tools = enabled_tools or []
        self.mcp_server_ids = mcp_server_ids or []
        self.history: List = []  # LangChain messages, carried across turns in this session
        self._model = ModelConfig.DEFAULT_MODEL
        _price = ModelConfig.get_pricing(self._model)
        self._input_price = _price["input"]
        self._output_price = _price["output"]

    # ── Persistence ──────────────────────────────────────────────────────
    async def _ensure_conversation(self, first_message: str) -> str:
        if self.conversation_id:
            await conversations_collection.update_one(
                {"_id": ObjectId(self.conversation_id)},
                {"$set": {"updated_at": datetime.now()}},
            )
            return self.conversation_id
        result = await conversations_collection.insert_one({
            "user_id": self.user_id,
            "title": first_message[:50],
            "modality": "voice",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })
        self.conversation_id = str(result.inserted_id)
        return self.conversation_id

    async def _persist_turn(self, transcript: str, response_text: str,
                             input_tokens: int, output_tokens: int, cost_usd: float,
                             timeline: Optional[list] = None, tool_steps: Optional[list] = None,
                             skills: Optional[list] = None, files_created: Optional[list] = None,
                             stop_reason: Optional[str] = None) -> None:
        cid = await self._ensure_conversation(transcript)
        await messages_collection.insert_one({
            "conversation_id": cid, "user_id": self.user_id,
            "role": "user", "content": transcript, "modality": "voice",
            "timestamp": datetime.now(),
        })
        # timeline/tool_steps/skills/files_created are written in the SAME shape
        # services/chat_service.py writes them for text turns — frontend
        # Message.jsx renders message.timeline (tool/skill/files nodes) and
        # falls back to plain content when it's absent, so a voice turn that
        # omitted these could never show its tool calls or its created files,
        # live or on reload. That was the "never shows the tool call like text
        # does" gap.
        doc = {
            "conversation_id": cid, "user_id": self.user_id,
            "role": "model", "content": response_text, "modality": "voice",
            "timeline": timeline or [], "tool_steps": tool_steps or [],
            "skills": skills or [], "files_created": files_created or [],
            "model": self._model, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 8), "timestamp": datetime.now(),
        }
        if stop_reason:
            doc["stopped"] = True
            doc["stop_reason"] = stop_reason
        await messages_collection.insert_one(doc)
        if cost_usd > 0:
            from utils.background_tasks import spawn
            spawn(CreditService.record_and_deduct(self.user_id, cost_usd), name="voice_credit_deduction")

    # ── Created-file detection ───────────────────────────────────────────
    # Same mechanism services/chat_service.py uses for text turns: snapshot
    # the conversation's outputs/ dir before the agent runs, diff mtimes
    # after. Without this a voice turn could create a real PDF and the user
    # would have no way to reach it — confirmed live: "create a simple pdf
    # for me" genuinely wrote outputs/simple_document.pdf and nothing about
    # it ever surfaced in the UI.
    async def _snapshot_outputs(self):
        from utils.workspace import conversation_workspace_for as _conv_ws_for
        outputs_dir = _conv_ws_for(self.user_id, self.conversation_id or "") / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        def _snap():
            return {
                f.name: f.stat().st_mtime
                for f in outputs_dir.iterdir()
                if f.is_file() and f.suffix.lower() in _CREATED_EXT
            }

        try:
            return await asyncio.to_thread(_snap), outputs_dir
        except Exception as e:  # noqa: BLE001 — never let this break a turn
            logger.warning("voice.pipeline.snapshot_failed", error=str(e))
            return {}, outputs_dir

    async def _detect_created_files(self, files_before: dict, outputs_dir) -> list:
        from utils import sandbox_client
        from utils.background_tasks import spawn

        # In remote sandbox mode the agent's code ran on the sandbox box, so
        # anything it wrote lives there — pull it down before diffing, same
        # ordering as chat_service.py's Step 7a.
        if sandbox_client.is_remote():
            try:
                await sandbox_client.sync_outputs(self.user_id, self.conversation_id, outputs_dir)
            except Exception as e:  # noqa: BLE001
                logger.error("voice.pipeline.output_sync_failed",
                             conversation_id=self.conversation_id, error=str(e))

        def _detect():
            out = []
            if not outputs_dir.exists():
                return out
            for f in sorted(outputs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if not f.is_file() or f.suffix.lower() not in _CREATED_EXT:
                    continue
                prev, curr = files_before.get(f.name), f.stat().st_mtime
                if prev is None or curr > prev:
                    out.append({
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                        "download_url": f"/outputs/my/{self.conversation_id}/{f.name}",
                        "ext": f.suffix.lower().lstrip("."),
                        "_path": str(f),
                    })
            return out

        created = []
        try:
            for item in await asyncio.to_thread(_detect):
                from services.chat_service import ChatService
                spawn(ChatService._bg_upload_to_cloudinary(
                    item.pop("_path"), item["name"], self.user_id, self.conversation_id),
                    name="voice_cloudinary_upload")
                created.append(item)
        except Exception as e:  # noqa: BLE001
            logger.warning("voice.pipeline.file_detection_failed", error=str(e))
        return created

    # ── Agent ────────────────────────────────────────────────────────────
    async def _run_agent_streaming(self, transcript: str) -> AsyncIterator[dict]:
        """Invokes the REAL agent graph (same one chat uses) and yields
        events as they stream:
            {"type": "text",       "content": str}
            {"type": "tool_start", "name", "args", "run_id"}
            {"type": "tool_end",   "name", "run_id", "result"}
            {"type": "skill",      "name", "content"}

        Tool events are surfaced (not just text) so voice reaches parity with
        text chat's tool timeline — see _persist_turn. Token usage is tracked
        the same way services/chat_service.py does, so voice turns get real
        cost accounting instead of a $0 placeholder."""
        agent_graph = await get_agent_graph()
        turn_id = str(uuid.uuid4())

        # Mirrors services/chat_service.py's mid-turn cost enforcement — the
        # pre-turn has_credit() check in run_utterance_from_stream only
        # blocks a NEW turn from starting; without this, an already-running
        # voice turn (e.g. a multi-tool-call loop) had no mechanism to stop
        # itself no matter how much it spent, unlike text chat.
        credit_is_exempt = await CreditService._is_admin(self.user_id)
        credit_spend_at_start = 0.0 if credit_is_exempt else await CreditService.get_spend(self.user_id)
        credit_cap = 0.0 if credit_is_exempt else await CreditService.get_cap(self.user_id)

        agent_input: ChatState = {
            "messages": [SystemMessage(content=VOICE_SYSTEM_PROMPT)] + self.history + [HumanMessage(content=transcript)],
            "user_id": self.user_id,
            "conversation_id": self.conversation_id or "",
            "enabled_tools": self.enabled_tools,
            "mcp_server_ids": self.mcp_server_ids,
            "selected_files": [],
        }
        config = {
            "run_name": f"voice | user={self.user_id[:8]}",
            "tags": [f"user:{self.user_id}", "voice"],
            "configurable": {
                "thread_id": self.conversation_id or turn_id,
                "enabled_tools": self.enabled_tools,
                "mcp_server_ids": self.mcp_server_ids,
                "user_id": self.user_id,
                "model": self._model,
                "turn_id": turn_id,
            },
            "recursion_limit": 150,
        }

        full_response = ""
        async for event in agent_graph.astream_events(agent_input, version="v2", config=config):
            if not isinstance(event, dict):
                continue
            etype = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node")

            if etype == "on_chat_model_start" and node == "agent_node":
                # A tool-using turn is several separate LLM messages (narrate
                # -> call tool -> narrate -> ...), and concatenating them raw
                # runs the sentences together: "...libraries available.I will
                # write a Python script..." — which reads wrong on screen and
                # makes the TTS run two sentences into one breath. Separate
                # messages with a space at the seam.
                if full_response and not full_response.endswith((" ", "\n")):
                    full_response += " "
                    yield {"type": "text", "content": " "}

            elif etype == "on_chat_model_stream" and node == "agent_node":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and isinstance(chunk.content, str) and chunk.content:
                    full_response += chunk.content
                    yield {"type": "text", "content": chunk.content}

            elif etype == "on_tool_start":
                tool_name = event.get("name")
                tool_args = event.get("data", {}).get("input")
                yield {"type": "tool_start", "name": tool_name,
                       "args": tool_args, "run_id": event.get("run_id")}
                if tool_name == "load_skill":
                    skill_name = (tool_args or {}).get("skill_name", "")
                    if skill_name:
                        try:
                            from skills.skill_loader import load_builtin_skill
                            skill_content = load_builtin_skill(skill_name) or ""
                        except Exception:  # noqa: BLE001
                            skill_content = ""
                        yield {"type": "skill", "name": skill_name, "content": skill_content}

            elif etype == "on_tool_end":
                yield {"type": "tool_end", "name": event.get("name"),
                       "run_id": event.get("run_id"),
                       "result": str(event.get("data", {}).get("output", ""))}

            elif etype == "on_chat_model_end" and node in ("agent_node", "agent_tool_node"):
                output_msg = event.get("data", {}).get("output")
                usage = getattr(output_msg, "usage_metadata", None) if output_msg else None
                if usage:
                    self._turn_input_tokens += usage.get("input_tokens", 0)
                    self._turn_output_tokens += usage.get("output_tokens", 0)

                    turn_cost_so_far = (self._turn_input_tokens * self._input_price
                                        + self._turn_output_tokens * self._output_price)
                    stop_reason = None
                    if not credit_is_exempt and credit_spend_at_start + turn_cost_so_far >= credit_cap + CREDIT_GRACE_USD:
                        logger.warning("voice.credit.grace_exceeded — stopping turn",
                                       user_id=self.user_id, turn_id=turn_id,
                                       spend_at_start=credit_spend_at_start,
                                       turn_cost_so_far=turn_cost_so_far, cap=credit_cap)
                        stop_reason = "credit_limit_reached"
                    elif turn_cost_so_far >= MAX_TURN_COST_USD:
                        logger.warning("voice.turn.cost_ceiling_exceeded — stopping turn",
                                       user_id=self.user_id, turn_id=turn_id,
                                       turn_cost_so_far=turn_cost_so_far, ceiling=MAX_TURN_COST_USD)
                        stop_reason = "turn_cost_ceiling"

                    if stop_reason:
                        # Yielded BEFORE cancel() so it's delivered to the
                        # client first — cancel() only takes effect at this
                        # generator's next await (the next astream_events
                        # fetch below), so the yield below is guaranteed to
                        # reach the caller first. Without this the turn would
                        # just go silent mid-sentence with no explanation,
                        # unlike text chat's distinct 'stopped' event.
                        yield {"type": "stopped", "reason": stop_reason}
                        asyncio.current_task().cancel()

        self.history.append(HumanMessage(content=transcript))
        if full_response:
            self.history.append(AIMessageChunk(content=full_response))

    # ── One-shot (REST STT) paths — simple, used by the offline test ──────
    async def process_utterance(self, audio_bytes: bytes, audio_content_type: str = "audio/wav") -> VoiceTurnResult:
        timing = {}
        t0 = time.monotonic()
        transcript = await transcribe_bytes(audio_bytes, content_type=audio_content_type)
        timing["stt_s"] = time.monotonic() - t0
        logger.info("voice.pipeline: transcribed", transcript=transcript, elapsed=timing["stt_s"])

        t1 = time.monotonic()
        response_text = ""
        audio_chunks = []
        first_llm_token_at = first_audio_byte_at = first_flush_at = None
        self._turn_input_tokens = 0
        self._turn_output_tokens = 0

        async def _text_stream():
            nonlocal response_text, first_llm_token_at
            async for ev in self._run_agent_streaming(transcript):
                if ev.get("type") != "text":
                    continue  # one-shot path has no UI to show tool steps in
                delta = ev["content"]
                if first_llm_token_at is None:
                    first_llm_token_at = time.monotonic()
                response_text += delta
                yield delta

        async for tts_chunk_text in chunk_for_tts(_text_stream()):
            if first_flush_at is None:
                first_flush_at = time.monotonic()

            async def _one(t=tts_chunk_text):
                yield t
            async for audio_chunk in synthesize_stream(_one()):
                if first_audio_byte_at is None:
                    first_audio_byte_at = time.monotonic()
                audio_chunks.append(audio_chunk)
        timing["agent_and_tts_s"] = time.monotonic() - t1

        if first_llm_token_at:
            timing["time_to_first_llm_token_s"] = first_llm_token_at - t1
        if first_flush_at:
            timing["time_to_first_sentence_flushed_s"] = first_flush_at - t1
        if first_audio_byte_at:
            timing["time_to_first_audio_byte_s"] = first_audio_byte_at - t1
        timing["total_s"] = timing["stt_s"] + timing["agent_and_tts_s"]

        cost_usd = self._turn_input_tokens * self._input_price + self._turn_output_tokens * self._output_price
        await self._persist_turn(transcript, response_text, self._turn_input_tokens, self._turn_output_tokens, cost_usd)

        return VoiceTurnResult(
            transcript=transcript, response_text=response_text,
            audio=b"".join(audio_chunks), timing=timing,
        )

    # ── Live streaming path — what the WebSocket route actually drives ────
    async def run_utterance_from_stream(self, audio_chunks: AsyncIterator[bytes],
                                         encoding: str = "linear16", sample_rate: int = 16000):
        """
        Real-time path: feeds a LIVE audio chunk iterator into Deepgram's
        streaming WebSocket API (interim results as the user talks), then —
        once Deepgram reports a final transcript (i.e. the caller's chunk
        iterator ends, signaling end of utterance) — runs the agent and
        streams TTS audio back exactly like process_utterance_streaming.

        Yields dicts:
          {"type": "transcript_partial", "text": str}
          {"type": "transcript_final", "text": str}
          {"type": "credit_blocked"}
          {"type": "tool_start", "name": str, "args": dict}
          {"type": "tool_end", "name": str}
          {"type": "audio_chunk", "data": bytes}
          {"type": "done", "response_text": str, "cost_usd": float,
           "timeline": list, "files_created": list, "conversation_id": str}
        """
        if not await CreditService.has_credit(self.user_id):
            yield {"type": "credit_blocked"}
            return

        final_transcript = ""
        async for event in stream_transcribe(audio_chunks, encoding=encoding, sample_rate=sample_rate):
            if event.is_final:
                final_transcript = (final_transcript + " " + event.text).strip() if final_transcript else event.text
                yield {"type": "transcript_final", "text": event.text}
            else:
                yield {"type": "transcript_partial", "text": event.text}

        if not final_transcript.strip():
            yield {"type": "done", "response_text": "", "cost_usd": 0.0,
                   "timeline": [], "files_created": [], "conversation_id": self.conversation_id}
            return

        # A conversation id is needed BEFORE the agent runs, because the
        # sandbox workspace (and therefore the outputs/ dir we diff for
        # created files) is scoped per conversation — see CLAUDE.md.
        await self._ensure_conversation(final_transcript)

        response_text = ""
        self._turn_input_tokens = 0
        self._turn_output_tokens = 0
        timeline: list = []
        tool_steps: list = []
        skills: list = []
        stop_reason: Optional[str] = None

        def _add_text(text: str) -> None:
            if timeline and timeline[-1].get("type") == "text":
                timeline[-1]["content"] += text
            else:
                timeline.append({"type": "text", "content": text})

        files_before, outputs_dir = await self._snapshot_outputs()
        files_created: list = []

        # Tool events have to reach the caller WHILE audio is streaming, but
        # audio is produced by a separate task (TurnSpeaker). Both are funneled
        # into one queue so this generator stays a single ordered event stream.
        merged: asyncio.Queue = asyncio.Queue()
        _AUDIO_DONE = object()

        async def _text_stream():
            nonlocal response_text, stop_reason, files_created
            async for ev in self._run_agent_streaming(final_transcript):
                etype = ev.get("type")
                if etype == "text":
                    response_text += ev["content"]
                    _add_text(ev["content"])
                    # Also pushed to the client as it streams, so the main chat
                    # renders a voice turn live exactly like a text turn instead
                    # of sitting empty until the whole turn finishes. Text runs
                    # slightly ahead of the spoken audio (TTS trails the model),
                    # which is the same behaviour every streaming voice UI has.
                    merged.put_nowait({"type": "text_delta", "content": ev["content"]})
                    yield ev["content"]
                    continue
                if etype == "tool_start":
                    step = {"name": ev["name"], "args": ev.get("args"),
                            "status": "running", "run_id": ev.get("run_id")}
                    tool_steps.append(step)
                    timeline.append({"type": "tool", **step})
                    merged.put_nowait({"type": "tool_start", "name": ev["name"], "args": ev.get("args")})
                elif etype == "tool_end":
                    run_id, name = ev.get("run_id"), ev.get("name")
                    for coll in (tool_steps, timeline):
                        for s in reversed(coll):
                            if s.get("run_id") == run_id or (
                                    s.get("name") == name and s.get("status") == "running"):
                                s["status"] = "completed"
                                s["result"] = ev.get("result", "")
                                break
                    # "result" was previously omitted here — persisted to
                    # tool_steps/timeline for reload, but never sent to the
                    # LIVE client, so a voice turn's tool card could only ever
                    # show "completed" with no output text (confirmed live:
                    # green checkmarks, code block, nothing else — exactly
                    # what omitting result produces in ChatPage.jsx's
                    # identical reducer, since it does {...tl[i], ...event.data}
                    # and there was no result key to spread in).
                    merged.put_nowait({"type": "tool_end", "name": name, "result": ev.get("result", "")})
                elif etype == "skill":
                    skill_data = {"name": ev["name"], "content": ev.get("content", "")}
                    skills.append(skill_data)
                    timeline.append({"type": "skill", **skill_data})
                    merged.put_nowait({"type": "skill", **skill_data})
                elif etype == "stopped":
                    stop_reason = ev.get("reason")
                    merged.put_nowait({"type": "stopped", "reason": stop_reason})

            # Agent + tool execution is fully done here, which can be well
            # before the trailing audio finishes — TTS lags behind text by
            # design (voice/speaker.py's burst model), so a response with a
            # few sentences after the file-creating tool call could leave
            # the UI sitting on an already-finished file for as long as
            # those sentences take to be spoken. Detect and emit it now
            # instead of waiting for the _AUDIO_DONE sentinel below.
            files_created = await self._detect_created_files(files_before, outputs_dir)
            if files_created:
                timeline.append({"type": "files_created", "files": files_created})
                merged.put_nowait({"type": "files_created", "files": files_created})

        # One Cartesia session per speech BURST, many bursts per turn — see
        # voice/speaker.py for why this must not be a single
        # synthesize_stream() call spanning the whole turn (short version:
        # a Cartesia context ends on idle, and letting that end cancel the
        # text iterator killed real in-flight tool calls mid-turn).
        speaker = TurnSpeaker(chunk_for_tts(_text_stream()), synthesize_stream=synthesize_stream)

        async def _pump_audio():
            try:
                async for audio in speaker.stream():
                    merged.put_nowait({"type": "audio_chunk", "data": audio})
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("voice.pipeline.audio_pump_failed",
                             error=f"{type(e).__name__}: {e}", exc_info=True)
            finally:
                merged.put_nowait(_AUDIO_DONE)

        audio_task = asyncio.create_task(_pump_audio())
        completed = False
        try:
            while True:
                item = await merged.get()
                if item is _AUDIO_DONE:
                    break
                yield item
            completed = True
        finally:
            if not audio_task.done():
                audio_task.cancel()
            await speaker.aclose()
            if not completed:
                # Interrupted (Stop pressed) or the client vanished mid-turn.
                # Save what was actually said anyway — the frontend has already
                # shown the user's transcript, so dropping it here would make a
                # stopped exchange disappear on reload. Spawned rather than
                # awaited because this path runs while the surrounding task is
                # being cancelled, and any await in that state re-raises
                # CancelledError immediately — a fire-and-forget task survives
                # it. Mirrors how chat_service.py persists a stopped text turn.
                #
                # NOTE on the cost-ceiling stop_reason set above: that
                # cancellation targets TurnSpeaker's internal pump task, which
                # is isolated by design (voice/speaker.py) — it does NOT
                # propagate to THIS generator, so a cost-ceiling stop is
                # actually caught by the fall-through "normal" persist path
                # below, not here. This branch only fires for a real external
                # interrupt (Stop button / client disconnect), where
                # stop_reason was never set by the cost check — hence the
                # explicit fallback.
                from utils.background_tasks import spawn
                partial_cost = (self._turn_input_tokens * self._input_price
                                + self._turn_output_tokens * self._output_price)
                spawn(self._persist_turn(final_transcript, response_text,
                                         self._turn_input_tokens, self._turn_output_tokens,
                                         partial_cost, timeline=timeline,
                                         tool_steps=tool_steps, skills=skills,
                                         files_created=files_created,
                                         stop_reason=stop_reason or "user_stop"),
                      name="voice_partial_persist")

        # files_created was already detected and emitted live above, right
        # as the agent's own work finished (not gated behind audio).
        cost_usd = self._turn_input_tokens * self._input_price + self._turn_output_tokens * self._output_price
        await self._persist_turn(final_transcript, response_text,
                                 self._turn_input_tokens, self._turn_output_tokens, cost_usd,
                                 timeline=timeline, tool_steps=tool_steps,
                                 skills=skills, files_created=files_created,
                                 stop_reason=stop_reason)

        yield {"type": "done", "response_text": response_text, "cost_usd": cost_usd,
               "timeline": timeline, "files_created": files_created,
               "stopped": bool(stop_reason), "stop_reason": stop_reason,
               # title matches what _ensure_conversation stored, so a client
               # adopting a backend-created conversation labels it correctly
               # in the sidebar without waiting for a refetch.
               "conversation_id": self.conversation_id, "title": final_transcript[:50]}
