"""
Sarvam text-to-speech client — raw WebSocket protocol, implemented directly
against the wire format because (like voice/tts.py's Cartesia client) full
control over exactly when text gets flushed is the whole point.

Wire format pulled from the ACTUAL sarvamai SDK source (raw_client.py /
socket_client.py / types/configure_connection_data.py, types/audio_output.py,
types/event_response.py under text_to_speech_streaming/) and confirmed live
against the real API — not from docs, which describe the SDK-wrapped call
shape rather than the literal JSON on the wire, and whose own example
speaker ("anushka") turned out to not even be valid for bulbul:v3 (confirmed
live: a real 422 from the API, which usefully listed the actual v3 roster).

  connect:  wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true
            header: Api-Subscription-Key
  1st msg:  {"type":"config","data":{"language_code","speaker","speech_sample_rate",
             "output_audio_codec","enable_preprocessing",...}}
  text:     {"type":"text","data":{"text": str}}
  flush:    {"type":"flush"}
  receive:  {"type":"audio","data":{"content_type","audio": base64, "request_id"}}
            {"type":"event","data":{"event_type":"final", ...}}
            {"type":"error","data":{...}}

One connection handles the WHOLE input (config once, many text messages,
one flush) — same "don't reconnect per sentence" principle voice/tts.py's
docstring explains, and the same reason voice/speaker.py's TurnSpeaker
exists: a provider-side idle/completion boundary must never be allowed to
end an agent TURN, only a SPEECH BURST.
"""
import base64
import json
import uuid
from typing import AsyncIterator

import structlog
import websockets

from voice import config

logger = structlog.get_logger(__name__)


"""NOTE: this module used to convert Sarvam's int16 output up to pcm_f32le
here, purely so the frontend could do `new Float32Array(buffer)` directly.
That convenience DOUBLED the bytes on the wire (768 kbps vs 384) on a link
measured at only ~414 kbps, and was a direct cause of the audio arriving at
a quarter of realtime. Sarvam is now asked for mu-law directly
(config.VOICE_WIRE_CODEC), which is 192 kbps, and its bytes are forwarded
untouched — no conversion step at all. See voice/config.py for the
measurements behind that choice."""


async def synthesize_stream(
    text_chunks: AsyncIterator[str],
    voice_id: str = None,
) -> AsyncIterator[bytes]:
    """Drop-in equivalent of voice.tts.synthesize_stream: streams text
    chunks to Sarvam and yields pcm_f32le audio bytes as they arrive.
    `voice_id` doubles as the Sarvam speaker name, kept as the same
    parameter name as the Cartesia version so pipeline.py/speaker.py don't
    need to know which provider they're talking to."""
    speaker = voice_id or config.SARVAM_TTS_SPEAKER
    url = (f"{config.SARVAM_TTS_WS_URL}?model={config.SARVAM_TTS_MODEL}"
           f"&send_completion_event=true")
    headers = {"Api-Subscription-Key": config.SARVAM_API_KEY}

    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "data": {
                "language_code": config.SARVAM_TTS_LANGUAGE_CODE,
                "speaker": speaker,
                "speech_sample_rate": config.SARVAM_TTS_SAMPLE_RATE,
                "output_audio_codec": config.VOICE_WIRE_CODEC,
                # Normalizes English words/numbers mixed into Hindi text —
                # on by default here since this pipeline's narration is
                # itself often code-mixed (model responding in Hindi with
                # English technical terms).
                "enable_preprocessing": True,
            },
        }))

        import asyncio

        # send_completion_event fires a "final" event PER FLUSH, not once
        # for the whole connection — confirmed live: 3 chunks each flushed
        # individually produced 3 separate final events, arriving right
        # after each chunk's own audio. Breaking on the first one (the
        # original version of this code, written before flush-per-chunk was
        # added) would have silently truncated every multi-sentence response
        # down to just its first sentence — worse than the stuttering this
        # change is meant to fix. So completion is now "every flush we sent
        # got its matching final back AND the sender has nothing left to
        # send", not "a final event arrived".
        flushes_sent = 0

        async def _sender():
            nonlocal flushes_sent
            sent_any = False
            async for text in text_chunks:
                if not text:
                    continue
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                # Flush after EVERY chunk, not just once at the end of the
                # burst. Sarvam buffers text server-side until it reaches
                # min_buffer_size (50 chars by default) OR sees a flush — our
                # own sentence/clause chunks (voice/sentence_chunker.py,
                # MIN_CHUNK_CHARS=20) are routinely under that threshold, so
                # without a flush per chunk, audio for an entire multi-
                # sentence burst could sit buffered server-side and arrive as
                # one late lump instead of streaming progressively. Confirmed
                # live as the actual cause of "small stops...voice cutting" —
                # this trades a few more small synthesis calls (TTS is
                # already faster than realtime, so that cost is free) for
                # audio that starts and continues on the same cadence the
                # text itself arrives.
                await ws.send(json.dumps({"type": "flush"}))
                flushes_sent += 1
                sent_any = True
            if not sent_any:
                # Nothing was ever said — close cleanly rather than hang
                # waiting for a "final" event that will never come, since
                # nothing was ever sent to trigger one.
                await ws.close()

        sender_task = asyncio.create_task(_sender())
        finals_received = 0

        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "audio":
                    # Forwarded verbatim — Sarvam is already producing the
                    # exact codec the browser decodes, so there is nothing
                    # to convert (and nothing to inflate).
                    audio = base64.b64decode(msg["data"]["audio"])
                    if audio:
                        yield audio
                elif mtype == "error":
                    raise RuntimeError(f"Sarvam TTS error: {msg}")
                elif mtype == "event":
                    if (msg.get("data") or {}).get("event_type") == "final":
                        finals_received += 1
                        if sender_task.done() and finals_received >= flushes_sent:
                            break
        except websockets.exceptions.ConnectionClosedOK:
            pass  # the "nothing was ever said" close() path above
        finally:
            if not sender_task.done():
                sender_task.cancel()


async def synthesize_text(text: str, voice_id: str = None) -> bytes:
    """Convenience wrapper matching voice.tts.synthesize_text's signature —
    synthesize one complete string, return all audio concatenated."""
    async def _one_shot():
        yield text

    chunks = []
    async for chunk in synthesize_stream(_one_shot(), voice_id=voice_id):
        chunks.append(chunk)
    return b"".join(chunks)
