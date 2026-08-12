"""
Cartesia text-to-speech client — raw WebSocket protocol (not the official
SDK), implemented directly against the wire format because that gives full
control over exactly when text gets flushed, which is the whole game for
low latency: never wait for a full LLM response before speaking, push each
sentence/clause the instant it's ready.

Wire format verified against Cartesia's own production usage in the
Pipecat framework (pipecat-ai/pipecat's cartesia/tts.py):
  connect:  wss://api.cartesia.ai/tts/websocket
            headers: X-API-Key, Cartesia-Version
  send:     {"transcript": str, "continue": bool, "context_id": str,
             "model_id": str, "voice": {"mode": "id", "id": str},
             "output_format": {"container", "encoding", "sample_rate"}}
  receive:  {"type": "chunk", "context_id": str, "data": base64 audio}
            {"type": "done", ...} / {"type": "error", ...}
"""
import base64
import json
import uuid
from typing import AsyncIterator

import structlog
import websockets

from voice import config

logger = structlog.get_logger(__name__)

# pcm_mulaw, not pcm_f32le: one wire format shared with the Sarvam client so
# the browser only ever decodes one thing, and 4x fewer bytes than float32 —
# see voice/config.py's VOICE_WIRE_CODEC for the bandwidth measurements that
# forced this change.
OUTPUT_FORMAT = {
    "container": "raw",
    "encoding": "pcm_mulaw",
    "sample_rate": config.CARTESIA_SAMPLE_RATE,
}


async def synthesize_stream(
    text_chunks: AsyncIterator[str],
    voice_id: str = None,
) -> AsyncIterator[bytes]:
    """
    Streams text chunks to Cartesia and yields raw PCM audio bytes as they
    arrive. Opens one WebSocket connection per call (simplest correct
    version — Cartesia's multi-context multiplexing over a single
    persistent connection is a later optimization, not needed to prove the
    pipeline works).
    """
    voice_id = voice_id or config.CARTESIA_VOICE_ID
    context_id = str(uuid.uuid4())

    url = f"{config.CARTESIA_WS_URL}?api_key={config.CARTESIA_API_KEY}&cartesia_version={config.CARTESIA_VERSION}"
    headers = {
        "X-API-Key": config.CARTESIA_API_KEY,
        "Cartesia-Version": config.CARTESIA_VERSION,
    }

    async with websockets.connect(
        config.CARTESIA_WS_URL,
        additional_headers=headers,
    ) as ws:

        def _msg(transcript: str, cont: bool) -> str:
            return json.dumps({
                "transcript": transcript,
                "continue": cont,
                "context_id": context_id,
                "model_id": config.CARTESIA_MODEL,
                "voice": {"mode": "id", "id": voice_id},
                "output_format": OUTPUT_FORMAT,
                "add_timestamps": False,
                "use_normalized_timestamps": False,
            })

        async def _sender():
            sent_any = False
            async for text in text_chunks:
                if not text:
                    continue
                await ws.send(_msg(text, True))
                sent_any = True
            # Signal no more text is coming — matches the official SDK's
            # ctx.no_more_inputs() as a distinct final message.
            await ws.send(_msg("" if sent_any else " ", False))

        import asyncio
        sender_task = asyncio.create_task(_sender())

        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "chunk" and msg.get("data"):
                    yield base64.b64decode(msg["data"])
                elif mtype == "error":
                    raise RuntimeError(f"Cartesia TTS error: {msg}")
                elif mtype == "done":
                    break
        finally:
            if not sender_task.done():
                sender_task.cancel()


async def synthesize_text(text: str, voice_id: str = None) -> bytes:
    """Convenience wrapper: synthesize one complete string, return all audio
    concatenated. Used for simple (non-incremental) callers/tests."""
    async def _one_shot():
        yield text

    chunks = []
    async for chunk in synthesize_stream(_one_shot(), voice_id=voice_id):
        chunks.append(chunk)
    return b"".join(chunks)
