"""
Sarvam speech-to-text client — raw WebSocket protocol, implemented directly
against the wire format (like voice/sarvam_tts.py) rather than pulling in
the full SDK, for the same reason: this pipeline only ever needs a handful
of specific calls, and a hand-rolled client makes exactly what's happening
inspectable.

Wire format pulled from the ACTUAL sarvamai SDK source
(speech_to_text_streaming/raw_client.py, socket_client.py,
types/audio_message.py, types/audio_data.py,
types/speech_to_text_transcription_data.py) and confirmed live — including
one real gotcha the docs don't mention: the connection-level sample_rate
query param only accepts 8000 or 16000 (a live 24000 attempt got an
immediate close, code 4000, "Unsupported sample rate: 24000. Supported
rates: 8000, 16000"), unlike the TTS side which does accept 24000.

  connect:  wss://api.sarvam.ai/speech-to-text/ws?language-code=hi-IN&model=saaras:v3&mode=codemix&sample_rate=16000
            header: Api-Subscription-Key
  send:     {"audio":{"data": base64, "sample_rate": int, "encoding":"audio/wav"}}
            {"type":"flush"}
  receive:  {"type":"data","data":{"request_id","transcript","language_code",...}}
            {"type":"error","data":{...}}
            {"type":"events","data":{...}}   (VAD signals, if enabled — unused here)

Key difference from Deepgram's stream_transcribe(): the AudioData message's
`encoding` field only accepts the literal "audio/wav", so each message sent
must be its own self-contained WAV file (44-byte header + PCM payload), not
a raw byte fragment of one continuous stream the way Deepgram's
send_media() works. Chunks are accumulated to ~600ms of audio per WAV
message before sending — short enough to keep latency reasonable, long
enough to give the ASR real acoustic context per call.

A "flush" is sent after EVERY accumulated chunk, not just once at the end
— this was NOT the original design and the first version shipped without
it, on the (wrong, unverified) assumption that sending audio alone would
produce incremental responses the way Deepgram streams. A live test
proved otherwise: an 8.7s utterance sent as ~13 separate WAV messages with
no flush in between produced ZERO transcript events until the FINAL flush
(end_utterance) — Sarvam was silently buffering every message server-side
and only responding once. So this endpoint's "streaming" is opt-in per
message via flush, not automatic from sending audio — matches the exact
lesson voice/sarvam_tts.py's per-chunk flush fix already established on
the TTS side of this pipeline, just discovered independently on the STT
side because the two failure modes look nothing alike (TTS: audio arrives
late in one lump; STT: nothing arrives until the very end at all).

The API does not appear to emit separate interim/final message types the
way Deepgram's Flux does (every "data" message is just "data" — no is_final
flag). So every "data" message here is treated as a completed SEGMENT
(is_final=True), the same way this module's Deepgram counterpart already
treats each genuine EndOfTurn — multi-segment accumulation in
voice/pipeline.py's run_utterance_from_stream() needs no change for this.
With the per-chunk flush above, a "data" response now arrives roughly once
per accumulation window (currently 1.8s — see _ACCUM_MS) WHILE the user is
still talking, which is what actually makes live progressive transcription
(as used by the dictation UI's growing preview text) possible at all.

DO NOT "fix" the resulting boundary corruption (a number or word split
across two chunks occasionally reads oddly) by redoing one unchunked pass
over the whole utterance at the end — this was tried and reverted after
live testing showed it's a worse failure mode, not a better one. A single
WAV message covering a real 4.78s two-sentence recording came back with
`metrics.audio_duration: 3.104` — Sarvam applies its own turn/pause
detection WITHIN a single message and silently drops audio after a
sufficiently long internal pause, independent of chunking strategy. That's
data loss, not just rough wording, and unlike a mangled number it can be
invisible to the user reviewing the result. routes/voice_routes.py's
/dictate endpoint therefore live-concatenates every chunked segment as the
authoritative final text, accepting fragmentation risk over data-loss risk.
"""
import asyncio
import base64
import io
import json
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import structlog
import websockets

from voice import config

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    confidence: float = 0.0


# How much audio to accumulate into one WAV-wrapped message (flushed
# immediately after sending — see _pump_audio) before starting the next one.
# Each accumulation window is transcribed roughly independently — confirmed
# live that Sarvam does NOT carry context across flushes within a
# connection, it transcribes only the new bytes since the last one. At the
# original 640ms, short/ambiguous content (spoken digits especially) got
# butchered: a real utterance reading out numbers came back as disjointed
# fragments ("Phillips 1", "5", "234", "4 * 8"...) instead of the correct
# sentence a single-pass transcription of the same audio produced cleanly.
# 1.8s is the tradeoff point chosen from that evidence: enough acoustic
# context per window to stop mangling short phrases, while still updating
# multiple times over any utterance longer than a few seconds — the actual
# reason this endpoint flushes per-chunk instead of once at the very end.
_ACCUM_MS = 1800


def _wrap_wav(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


async def stream_transcribe(
    audio_chunks: AsyncIterator[bytes],
    encoding: str = "linear16",
    sample_rate: int = 16000,
) -> AsyncIterator[TranscriptEvent]:
    if sample_rate not in (8000, 16000):
        raise ValueError(f"Sarvam STT only supports 8000/16000 Hz, got {sample_rate}")

    url = (f"{config.SARVAM_STT_WS_URL}?language-code={config.SARVAM_STT_LANGUAGE_CODE}"
           f"&model={config.SARVAM_STT_MODEL}&mode={config.SARVAM_STT_MODE}"
           f"&sample_rate={sample_rate}")
    headers = {"Api-Subscription-Key": config.SARVAM_API_KEY}

    accum_bytes = int(sample_rate * 2 * _ACCUM_MS / 1000)  # 2 bytes/sample (int16)
    queue: asyncio.Queue = asyncio.Queue()
    # A server-side error (bad auth, quota, malformed request) used to only
    # get logged — the caller saw zero transcript events and no indication
    # anything went wrong, which is indistinguishable from the user simply
    # not having said anything. Stored here and raised after the loop ends,
    # so it propagates out of stream_transcribe into
    # run_utterance_from_stream's un-wrapped `async for`, up to
    # routes/voice_routes.py's existing generic exception handler — reusing
    # the error-surfacing path that's already there and already tested,
    # rather than building new plumbing for this one case.
    server_error: Optional[str] = None

    async with websockets.connect(url, additional_headers=headers) as ws:

        async def _listener():
            nonlocal server_error
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "data":
                        text = (msg.get("data") or {}).get("transcript") or ""
                        if text:
                            queue.put_nowait(TranscriptEvent(text=text, is_final=True))
                    elif mtype == "error":
                        server_error = str(msg)
                        logger.warning(f"voice.sarvam_stt: server error: {msg}")
            except websockets.exceptions.ConnectionClosed:
                pass

        async def _pump_audio():
            # A "flush" after EVERY accumulated chunk, not just once at the
            # end — confirmed live this is required: without it, Sarvam's
            # STT WS silently buffers every audio message server-side and
            # only responds once, right after the FINAL flush, no matter how
            # many WAV-wrapped messages were sent along the way. A live test
            # that streamed an 8.7s utterance in ~640ms chunks got exactly
            # ZERO transcript events until end_utterance's flush — the
            # accumulate-then-send-without-flush version of this function
            # was accidentally a one-shot batch transcriber wearing a
            # streaming API's clothes, not the incremental live transcript
            # this endpoint exists to provide. Same lesson as
            # voice/sarvam_tts.py's per-chunk flush fix, just on the other
            # leg of the pipeline: this provider's "streaming" is opt-in per
            # message via flush, not automatic from sending audio alone.
            buf = bytearray()
            async for chunk in audio_chunks:
                buf.extend(chunk)
                if len(buf) >= accum_bytes:
                    wav = _wrap_wav(bytes(buf), sample_rate)
                    await ws.send(json.dumps({
                        "audio": {"data": base64.b64encode(wav).decode(),
                                  "sample_rate": sample_rate, "encoding": "audio/wav"},
                    }))
                    await ws.send(json.dumps({"type": "flush"}))
                    buf.clear()
            if buf:
                wav = _wrap_wav(bytes(buf), sample_rate)
                await ws.send(json.dumps({
                    "audio": {"data": base64.b64encode(wav).decode(),
                              "sample_rate": sample_rate, "encoding": "audio/wav"},
                }))
            await ws.send(json.dumps({"type": "flush"}))

        listener_task = asyncio.create_task(_listener())
        pump_task = asyncio.create_task(_pump_audio())

        try:
            while True:
                # Stop as soon as the server reports an error rather than
                # waiting out the full pump — there's nothing left to
                # transcribe once the connection has told us it's broken.
                if server_error is not None:
                    break
                pump_done = pump_task.done()
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield event
                except asyncio.TimeoutError:
                    if pump_done and queue.empty():
                        break
            # A brief grace window for the server's response to the final
            # flush to arrive after our own audio stream has ended — the
            # same "don't just walk away right after our last send" allowance
            # voice/stt.py's Deepgram version needs, since transcription of
            # the tail end of an utterance isn't instantaneous.
            try:
                while True:
                    event = await asyncio.wait_for(queue.get(), timeout=1.5)
                    yield event
            except asyncio.TimeoutError:
                pass
            if server_error is not None:
                raise RuntimeError(f"Sarvam STT error: {server_error}")
        finally:
            for t in (pump_task, listener_task):
                if not t.done():
                    t.cancel()
