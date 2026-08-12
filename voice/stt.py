"""
Deepgram speech-to-text client.

Two entry points:
  - transcribe_bytes(): one-shot REST call. Simple, reliable, ~0.5s round
    trip for a 10s clip (measured live against the real API). Good fallback
    and good enough for a "record until silence, then transcribe" flow.
  - stream_transcribe(): real WebSocket streaming with interim results —
    the actual low-latency path a live voice agent needs, since it lets the
    agent start reacting before the user finishes speaking.
"""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
import structlog

from voice import config

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    confidence: float = 0.0


async def transcribe_bytes(audio_bytes: bytes, content_type: str = "audio/wav") -> str:
    """One-shot transcription via Deepgram's REST /listen endpoint. Returns
    the plain transcript text. Raises on non-200 (caller decides fallback)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{config.DEEPGRAM_REST_URL}?model={config.DEEPGRAM_MODEL}&smart_format=true",
            headers={
                "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
                "Content-Type": content_type,
            },
            content=audio_bytes,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Deepgram REST transcription failed: {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    return data["results"]["channels"][0]["alternatives"][0]["transcript"]


async def stream_transcribe(
    audio_chunks: AsyncIterator[bytes],
    encoding: str = "linear16",
    sample_rate: int = 16000,
) -> AsyncIterator[TranscriptEvent]:
    """
    Streams raw PCM audio chunks to Deepgram's live /v2/listen ("Flux")
    WebSocket API and yields TranscriptEvent as interim/final results
    arrive. Uses the deepgram-sdk directly rather than hand-rolling the WS
    wire protocol (Deepgram's live API has keepalive/reconnect semantics
    best left to the maintained SDK).

    NOTE on the real (SDK-verified, not doc-guessed) wire shapes here — all
    confirmed live: an initial doc-page fetch 404'd, so the model name and
    socket-client method names below came from a WebFetch summary of
    Pipecat's Deepgram integration and turned out to be subtly wrong twice
    in a row before this settled (nova-3 rejected on /v2, then send()/
    finish() didn't exist as methods, then send_media()/send_close_stream()
    turned out to be coroutines needing await, then finally — the one that
    silently dropped every event with zero errors — MESSAGE callbacks
    receive plain dicts, NOT the SDK's ListenV2TurnInfo pydantic objects the
    type hints suggest, so attribute access (message.transcript) always
    raises/returns nothing and get()-style dict access is required):
      - model MUST be "flux-general-en"/"flux-general-multi" — NOT a
        REST/v1 model like "nova-3" (v1 and v2 are unrelated model
        families; nova-3 on v2 gets a bare "HTTP 400: Unexpected error
        when initializing websocket connection" with no further detail).
      - the socket client's send/close methods are the coroutines
        `await send_media(bytes)` / `await send_close_stream()` — NOT
        send()/finish() (don't exist) and NOT fire-and-forget (both must
        be awaited or the bytes silently never send).
      - MESSAGE events are plain dicts shaped like the TurnInfo schema
        (`{"type": "TurnInfo", "event": "Update"|"StartOfTurn"|
        "EagerEndOfTurn"|"TurnResumed"|"EndOfTurn", "transcript": str,
        "end_of_turn_confidence": float, ...}`) — NOT nested
        `.channel.alternatives[0]` like the v1 REST shape, and not
        attribute-accessible despite the SDK's own type hints.
      - confirmed live: EndOfTurn does NOT reliably fire within a short
        push-to-talk window (Flux's own turn-detection has its own timeout
        independent of when the caller stops sending audio) — every event
        in a 4.4s test utterance came back "Update"/"StartOfTurn", never
        "EndOfTurn", even after send_close_stream(). So finality here is
        NOT "Deepgram said EndOfTurn" — it's "the caller's own audio
        stream ended", and this function synthesizes one final event from
        the last transcript seen once the pump loop finishes, rather than
        waiting for a signal that may never arrive in time.
    """
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType

    client = AsyncDeepgramClient(api_key=config.DEEPGRAM_API_KEY)
    queue: asyncio.Queue = asyncio.Queue()
    # Tracks whether the MOST RECENT transcript text has already been
    # emitted as final — not a single stream-wide flag. A push-to-talk
    # utterance can legitimately contain several genuine EndOfTurn events
    # (Flux advances turn_index on its own internal pauses), and content
    # spoken AFTER the last real EndOfTurn but before the caller's own
    # audio stream ends must still reach the caller. Confirmed live: a
    # single global "already got one final, never synthesize again" flag
    # silently dropped the second half of a two-clause test utterance
    # ("Keep your answer short.") before it ever reached the agent.
    last_transcript = ""
    last_transcript_is_final = False
    # A server-side error (bad auth, quota, malformed request) used to only
    # get logged — the caller saw zero transcript events and no indication
    # anything went wrong, indistinguishable from the user simply not having
    # said anything. Raised after the loop ends so it propagates out of
    # stream_transcribe into run_utterance_from_stream's un-wrapped
    # `async for`, reaching routes/voice_routes.py's existing generic
    # exception handler for free.
    server_error = None

    async with client.listen.v2.connect(
        model=config.DEEPGRAM_STREAM_MODEL,
        encoding=encoding,
        sample_rate=sample_rate,
    ) as connection:

        def on_error(err):
            nonlocal server_error
            server_error = str(err)
            logger.error(f"voice.stt: Deepgram error: {err}")

        def on_message(message):
            nonlocal last_transcript, last_transcript_is_final
            try:
                if not isinstance(message, dict) or message.get("type") != "TurnInfo":
                    return  # ignore Connected/ConfigureSuccess/etc — not a transcript
                text = message.get("transcript") or ""
                is_final = message.get("event") == "EndOfTurn"
                if text:
                    last_transcript = text
                    last_transcript_is_final = is_final
                    queue.put_nowait(TranscriptEvent(
                        text=text, is_final=is_final,
                        confidence=message.get("end_of_turn_confidence") or 0.0,
                    ))
            except Exception as e:  # noqa: BLE001 — never let a malformed event kill the stream
                logger.warning(f"voice.stt: could not parse Deepgram message: {e}")

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        listen_task = asyncio.create_task(connection.start_listening())

        async def _pump_audio():
            async for chunk in audio_chunks:
                await connection.send_media(chunk)
            await connection.send_close_stream()  # signal end-of-audio, flush final transcript

        pump_task = asyncio.create_task(_pump_audio())

        try:
            while True:
                if server_error is not None:
                    break
                pump_done = pump_task.done()
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield event
                except asyncio.TimeoutError:
                    if pump_done and queue.empty():
                        break
            # Our own audio stream ended with trailing content that never
            # got a real EndOfTurn (whether this is the whole utterance or
            # just its last clause after an earlier genuine final — see the
            # module-level note above) — synthesize a final for exactly
            # that trailing piece, not the whole utterance.
            if not last_transcript_is_final and last_transcript:
                yield TranscriptEvent(text=last_transcript, is_final=True)
            if server_error is not None:
                raise RuntimeError(f"Deepgram STT error: {server_error}")
        finally:
            for t in (pump_task, listen_task):
                if not t.done():
                    t.cancel()
