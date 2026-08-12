"""
Splits a stream of LLM token deltas into TTS-flushable chunks on natural
pause points (sentence/clause boundaries), instead of waiting for the full
response. This is the actual latency trick every cascaded voice agent
relies on — Kokoro/Cartesia/any TTS engine's own "fast" benchmark numbers
are meaningless if you feed them one giant paragraph at once (confirmed
live earlier this session: 6.9s to first audio for one long sentence vs
~1-2.6s for short clauses on the same engine).
"""
import re
from typing import AsyncIterator

# Sentence-ending punctuation, or a comma followed by whitespace (clause
# boundary) once a chunk has grown past MIN_CHUNK_CHARS — flushing on every
# single comma would fragment audio too much and hurt prosody.
_SENTENCE_END = re.compile(r'[.!?]+["\')\]]?\s')
_CLAUSE_END = re.compile(r',\s')

MIN_CHUNK_CHARS = 20


async def chunk_for_tts(token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Accumulates streamed text and yields complete flushable chunks as
    soon as a sentence boundary appears, or a clause boundary once the
    buffer is already long enough to be worth speaking on its own."""
    buffer = ""
    async for token in token_stream:
        buffer += token

        while True:
            m = _SENTENCE_END.search(buffer)
            if m:
                yield buffer[:m.end()].strip()
                buffer = buffer[m.end():]
                continue
            if len(buffer) >= MIN_CHUNK_CHARS:
                m = _CLAUSE_END.search(buffer)
                if m:
                    yield buffer[:m.end()].strip()
                    buffer = buffer[m.end():]
                    continue
            break

    if buffer.strip():
        yield buffer.strip()
