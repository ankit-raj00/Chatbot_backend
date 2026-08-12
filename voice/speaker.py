"""
Drives Cartesia TTS across a WHOLE agent turn, rather than treating the
turn as one continuous stretch of speech.

Why this module exists
----------------------
voice/tts.py's synthesize_stream() is a single Cartesia "context", and a
context ENDS (Cartesia sends {"type": "done"}) once it has spoken
everything it was given and no further text shows up. That is exactly
right for one continuous burst of speech — and exactly wrong for an agent
turn, because an agent turn is not one burst: it narrates a sentence,
then goes silent for 5-30 seconds running a tool, then narrates the next
one.

Feeding a whole turn through one synthesize_stream() call therefore ends
the TURN the moment the first tool call runs long enough for Cartesia to
close the context: synthesize_stream() returns, its finally block cancels
the task that was iterating the text, and that cancellation propagates
down the iterator chain (chunk_for_tts -> the agent's token stream ->
astream_events) and kills real in-flight tool calls.

Confirmed live before this fix: a "create a simple pdf for me" turn ran
load_skill plus three sandbox_run_python calls correctly, then
sandbox_analyze_image logged tool.pre_call at 09:26:55.556 and NEVER
logged tool.post_call — while the turn was persisted 1.2s later at
09:26:56.726 containing only the narration lines ("I will load the PDF
creation skill...I will analyze the rendered image...") and no actual
answer. The graph cannot have finished on its own, since a finished graph
means every tool call returned; it was cancelled from underneath.

The fix: one Cartesia session per BURST, many bursts per turn. A gap
longer than BURST_IDLE_S means the agent is busy working, so the session
is closed deliberately (on OUR schedule, not by racing Cartesia's own
idle policy) and a fresh one opens when speech resumes. The reconnect
cost lands inside a gap that already exists, so it is inaudible — unlike
the earlier per-SENTENCE reconnection bug, which put a handshake between
consecutive sentences of the same breath and was very audible.

The turn now ends when the AGENT's text stream is exhausted, never when
the TTS provider decides it has nothing left to say.
"""
import asyncio
from typing import AsyncIterator, Optional

import structlog

from voice.tts import synthesize_stream as _default_synthesize_stream

logger = structlog.get_logger(__name__)

# A silence longer than this means the agent is working (tool call), not
# pausing mid-thought — so end the current Cartesia session rather than
# holding an idle context open and letting the provider decide.
BURST_IDLE_S = 1.5

_SENTINEL = object()


class TurnSpeaker:
    """Turns a stream of ready-to-speak sentences (chunk_for_tts output)
    into a stream of audio bytes, surviving arbitrarily long silent gaps
    in the middle of a turn."""

    def __init__(self, sentences: AsyncIterator[str], voice_id: Optional[str] = None,
                 idle_s: float = BURST_IDLE_S, synthesize_stream=None):
        self._source = sentences
        self._voice_id = voice_id
        self._idle_s = idle_s
        # Defaults to Cartesia for any other caller of this class, but
        # pipeline.py passes its own provider-resolved synthesize_stream
        # (Sarvam or Cartesia, per VOICE_TTS_PROVIDER) so TurnSpeaker never
        # has to hardcode which TTS backend it's fronting.
        self._synthesize_stream = synthesize_stream or _default_synthesize_stream
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pump: Optional[asyncio.Task] = None
        self._exhausted = False

    async def _run_pump(self) -> None:
        """Consumes the sentence source in its own task, so that a Cartesia
        session ending can never cancel the agent producing the text —
        that decoupling is the entire point of this class."""
        try:
            async for sentence in self._source:
                await self._queue.put(sentence)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a failed agent must not hang the speaker
            logger.error("voice.speaker.pump_failed",
                         error=f"{type(e).__name__}: {e}", exc_info=True)
        finally:
            await self._queue.put(_SENTINEL)

    async def _take(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._exhausted:
            return None
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if item is _SENTINEL:
            self._exhausted = True
            return None
        return item

    async def _burst(self, first: str) -> AsyncIterator[str]:
        """One Cartesia session's worth of text: everything that keeps
        arriving promptly. Ends on an idle gap (agent went off to run a
        tool) or on the source being exhausted."""
        yield first
        while True:
            try:
                nxt = await self._take(timeout=self._idle_s)
            except asyncio.TimeoutError:
                return  # agent is working — end this session, more may follow
            if nxt is None:
                return
            yield nxt

    async def stream(self) -> AsyncIterator[bytes]:
        self._pump = asyncio.create_task(self._run_pump())
        while True:
            # No timeout here: this is the wait that spans a tool call, and
            # waiting is correct — the turn is not over until the agent's
            # stream actually ends.
            first = await self._take()
            if first is None:
                return
            async for audio in self._synthesize_stream(self._burst(first), voice_id=self._voice_id):
                yield audio

    async def aclose(self) -> None:
        """Stops the agent-side pump. Called when the turn is torn down for
        real (client disconnect), which is the one case where cancelling the
        agent IS the intended behaviour."""
        if self._pump and not self._pump.done():
            self._pump.cancel()
            try:
                await self._pump
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
