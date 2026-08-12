"""
WebSocket transport for the voice pipeline. Deliberately thin — auth, then
relay: binary frames in are audio chunks, JSON frames in are control
messages, JSON/binary frames out are pipeline events. All real logic (STT,
agent, TTS, credit gating, persistence) lives in voice/pipeline.py.
"""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from bson import ObjectId

from core.auth import verify_token
from core.database import users_collection
from voice.pipeline import VoicePipeline, stream_transcribe
from voice import config as voice_config

import structlog
logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice"])

# Sentinel put on the per-utterance audio queue to signal "no more audio for
# this utterance" — lets the consuming async generator terminate cleanly
# instead of blocking forever on queue.get().
_END_OF_UTTERANCE = object()


async def _authenticate_ws(websocket: WebSocket):
    """Mirrors core/middleware.py's get_current_user, adapted for a
    WebSocket handshake: Starlette's WebSocket exposes .cookies the same
    way Request does (the access_token cookie is SameSite=None; Secure in
    prod — see controllers/auth_controller.py — so it's sent on a
    cross-origin WS handshake the same way it is on a normal XHR/fetch).
    A ?token= query param is accepted as a fallback for any environment
    that strips cookies on the WS upgrade.
    """
    token = websocket.cookies.get("access_token") or websocket.query_params.get("token")
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        return None
    user_id = payload.get("user_id")
    if not user_id:
        return None
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    return user


@router.websocket("/ws")
async def voice_ws(
    websocket: WebSocket,
    conversation_id: str | None = Query(default=None),
    # Same tool/MCP selection the frontend already sends on every chat turn
    # (ChatPage.jsx's selectedTools/selectedMcpServers) — comma-separated
    # since WS query params don't have a native list encoding the way a
    # JSON POST body does. Empty/missing means "sandbox + skills only",
    # same as before this was wired through.
    enabled_tools: str = Query(default=""),
    mcp_server_ids: str = Query(default=""),
):
    user = await _authenticate_ws(websocket)
    if not user:
        await websocket.close(code=4401, reason="unauthorized")
        return

    user_id = str(user["_id"])
    parsed_enabled_tools = [t for t in enabled_tools.split(",") if t]
    parsed_mcp_server_ids = [m for m in mcp_server_ids.split(",") if m]
    await websocket.accept()
    logger.info("voice.ws.connected", user_id=user_id, conversation_id=conversation_id,
                enabled_tools=parsed_enabled_tools, mcp_server_ids=parsed_mcp_server_ids)

    try:
        pipeline = VoicePipeline(
            user_id=user_id, conversation_id=conversation_id,
            enabled_tools=parsed_enabled_tools, mcp_server_ids=parsed_mcp_server_ids,
        )
    except RuntimeError as e:
        # voice_config.require_keys() failed — provider keys not configured.
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1011, reason="voice pipeline unavailable")
        return

    audio_queue: asyncio.Queue | None = None
    turn_task: asyncio.Task | None = None

    async def _queue_to_chunks(queue: asyncio.Queue):
        # Takes the queue as an ARGUMENT rather than closing over the
        # `audio_queue` local: a closure reads the variable's CURRENT value
        # at call time, so clearing audio_queue = None at the end of an
        # utterance would make an already-running turn's generator blow up
        # with AttributeError on None.get().
        while True:
            item = await queue.get()
            if item is _END_OF_UTTERANCE:
                return
            yield item

    async def _run_turn(queue: asyncio.Queue):
        # Held explicitly (not just the bare `async for` target) so the
        # finally block below can force-close it. Confirmed live: when the
        # client disconnects WHILE this loop is mid-send (send_bytes/
        # send_json raises WebSocketDisconnect), the except block below
        # catches that and returns — but returning out of an `async for`
        # does NOT itself stop the generator being iterated. The agent
        # graph underneath (real tool calls: pip installs, sandbox code,
        # ffmpeg downloads) kept running to completion in the background
        # with nobody listening, since nothing ever told it to stop -
        # confirmed by tool.pre_call/tool.post_call log lines continuing
        # for a full multi-tool turn AFTER voice.ws.disconnected had
        # already logged. gen.aclose() in finally throws GeneratorExit
        # into the generator at wherever it's currently suspended, which
        # propagates down through stream_transcribe/synthesize_stream/the
        # agent graph's own astream_events and actually stops the work.
        gen = pipeline.run_utterance_from_stream(_queue_to_chunks(queue))
        try:
            async for event in gen:
                etype = event.get("type")
                if etype == "audio_chunk":
                    await websocket.send_bytes(event["data"])
                else:
                    await websocket.send_json(event)
        except asyncio.CancelledError:
            await websocket.send_json({"type": "interrupted"})
            raise
        except Exception as e:  # noqa: BLE001 — never let one bad turn kill the socket
            # str(e) alone can render blank for some exception types (e.g.
            # asyncio.CancelledError-adjacent ones) — always include the
            # type name too, so a failure is never a silent empty string.
            logger.error("voice.ws.turn_failed", user_id=user_id,
                         error=f"{type(e).__name__}: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "error", "message": "Something went wrong processing that."})
            except Exception:
                pass
        finally:
            # See the comment above gen's assignment — this is the actual
            # fix, not the except blocks. Safe/idempotent if the generator
            # already ran to completion normally.
            await gen.aclose()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                if audio_queue is None:
                    # First audio chunk of a new utterance — implicitly starts it.
                    # If the previous turn is somehow still going (user talked
                    # over the response), stop it rather than running two turns
                    # concurrently and interleaving two voices into one socket.
                    if turn_task and not turn_task.done():
                        turn_task.cancel()
                        # AWAITED, not fire-and-forget: _run_turn writes to
                        # THIS websocket via send_bytes/send_json, and
                        # starting a new turn_task immediately would let the
                        # old task's in-flight send() (its 'interrupted'
                        # message, or whatever frame it was mid-write on)
                        # race the new task's first send() on the same
                        # WebSocket — concurrent writers on one WS connection
                        # is not a safe pattern (frame interleaving / a
                        # "cannot call send once a close message has been
                        # sent" crash if the timing lands wrong). cancel()
                        # only SCHEDULES the exception for the old task's next
                        # await point; only awaiting it here guarantees that
                        # task has actually stopped writing before a new one
                        # starts. The old task's own CancelledError handler
                        # already suppresses/re-raises appropriately, so this
                        # is just draining that, not duplicating its logic.
                        try:
                            await turn_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                    audio_queue = asyncio.Queue()
                    turn_task = asyncio.create_task(_run_turn(audio_queue))
                await audio_queue.put(message["bytes"])
                continue

            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                ctype = control.get("type")

                if ctype == "end_utterance":
                    if audio_queue is not None:
                        # Deliberately NOT awaited: this loop is the ONLY
                        # reader of the socket, so blocking here for the whole
                        # turn (often 30-60s with tool calls) meant an
                        # "interrupt" control message — the Stop button —
                        # could not be read until the turn it was meant to
                        # stop had already finished. The turn task is left
                        # running and cleaned up in the finally block below;
                        # it owns its own queue, so clearing these is safe.
                        await audio_queue.put(_END_OF_UTTERANCE)
                        audio_queue = None
                    else:
                        # The client stopped recording without ever sending
                        # audio (a tap too short to fill one capture buffer),
                        # so no turn exists. Say so explicitly — otherwise the
                        # client sits in "Thinking…" forever waiting for a
                        # response that nothing is going to produce.
                        await websocket.send_json({"type": "done", "response_text": "", "cost_usd": 0.0})

                elif ctype == "interrupt":
                    if turn_task and not turn_task.done():
                        turn_task.cancel()
                        # Awaited for the same reason as the barge-in path
                        # above: guarantees the cancelled task has stopped
                        # writing to this websocket (including its own
                        # 'interrupted' send) before turn_task is cleared to
                        # None and the loop goes back to accepting whatever
                        # the client sends next — including audio bytes that
                        # would otherwise start a NEW task able to write
                        # concurrently with this one's still-unwinding send().
                        try:
                            await turn_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                    if audio_queue is not None:
                        await audio_queue.put(_END_OF_UTTERANCE)
                    audio_queue = None
                    turn_task = None

    except WebSocketDisconnect:
        pass
    finally:
        if turn_task and not turn_task.done():
            turn_task.cancel()
        logger.info("voice.ws.disconnected", user_id=user_id)


@router.websocket("/dictate")
async def dictate_ws(websocket: WebSocket):
    """
    Voice-to-TEXT only — no agent, no TTS, no credit gating. Deliberately a
    separate, much simpler endpoint rather than a mode flag on /voice/ws:
    dictation has none of that endpoint's real complexity (turn lifecycle,
    barge-in, TTS bursts, cost ceilings) and bolting a flag onto it would
    make the already-nontrivial /voice/ws harder to reason about for a
    feature that doesn't need any of it. Single utterance per connection —
    the client connects, streams audio, sends end_utterance, gets the
    transcript, and the connection closes; a fresh dictation opens a fresh
    connection, matching the simple "press mic, speak, press again" UX this
    is actually for (no session to manage on either side).

    No credit check: STT-only has no LLM/TTS cost, so there is nothing to
    gate — this uses provider quota, not the user's chat credit.
    """
    user = await _authenticate_ws(websocket)
    if not user:
        await websocket.close(code=4401, reason="unauthorized")
        return
    user_id = str(user["_id"])

    try:
        voice_config.require_stt_key()
    except RuntimeError as e:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1011, reason="voice pipeline unavailable")
        return

    await websocket.accept()
    logger.info("voice.dictate.connected", user_id=user_id)

    audio_queue: asyncio.Queue = asyncio.Queue()

    async def _chunks():
        while True:
            item = await audio_queue.get()
            if item is _END_OF_UTTERANCE:
                return
            yield item

    async def _run():
        # Deliberately NOT "redo one clean pass over the whole utterance" —
        # that was tried and reverted after live testing showed it silently
        # DROPS content, not just garbles it. Confirmed directly: a single
        # unchunked WAV covering a real 4.78s two-sentence recording came
        # back with metrics.audio_duration=3.104 — Sarvam's own turn/VAD
        # detection stopped at the first sentence-length pause and never
        # transcribed the rest, independent of how the audio was chunked or
        # flushed on the way in. Silent truncation is a strictly worse
        # failure mode than the localized word/number-boundary corruption
        # the live chunked concatenation below can produce (see
        # voice/sarvam_stt.py's _ACCUM_MS comment) — a slightly mangled
        # number is visibly wrong and correctable before sending; a
        # cleanly-worded sentence that's just missing its second half looks
        # completely fine and might not even be noticed. So this endpoint
        # accepts fragmentation risk over data-loss risk, live-concatenating
        # every transcript_final segment as the actual final answer.
        final_transcript = ""
        try:
            async for event in stream_transcribe(_chunks()):
                if event.is_final:
                    final_transcript = (final_transcript + " " + event.text).strip() if final_transcript else event.text
                    await websocket.send_json({"type": "transcript_final", "text": event.text})
                else:
                    await websocket.send_json({"type": "transcript_partial", "text": event.text})
            await websocket.send_json({"type": "done", "text": final_transcript})
        except Exception as e:  # noqa: BLE001 — same posture as voice_ws: never let one bad utterance kill the socket silently
            logger.error("voice.dictate.failed", user_id=user_id,
                         error=f"{type(e).__name__}: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "error", "message": "Something went wrong transcribing that."})
            except Exception:
                pass

    run_task: asyncio.Task | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                if run_task is None:
                    run_task = asyncio.create_task(_run())
                await audio_queue.put(message["bytes"])
                continue
            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "end_utterance":
                    await audio_queue.put(_END_OF_UTTERANCE)
                    if run_task:
                        await run_task
                    break  # single utterance per connection — done
    except WebSocketDisconnect:
        pass
    finally:
        if run_task and not run_task.done():
            run_task.cancel()
        logger.info("voice.dictate.disconnected", user_id=user_id)
