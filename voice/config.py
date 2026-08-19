"""Env-driven config for the voice pipeline — mirrors the pattern used
throughout this codebase (e.g. rag/parsers/parser_client.py) of module-level
constants read once from os.getenv with sane defaults."""
import os

import structlog

logger = structlog.get_logger(__name__)

# ── Provider selection ──────────────────────────────────────────────────
# Sarvam is the default for both legs: it fixed a real accuracy complaint on
# Indian-accented speech (Deepgram Flux was mishearing it) and — verified
# live via a real TTS->STT round trip — correctly handles code-mixed
# Hindi/English ("Hinglish") with the meaning fully intact, which neither
# Deepgram nor Cartesia can do at all. Deepgram/Cartesia code is left in
# place and selectable via env, not deleted — there was nothing wrong with
# them technically (Cartesia in particular measured ~2x lower TTS latency
# live), the switch is about matching the actual target users' speech.
STT_PROVIDER = os.getenv("VOICE_STT_PROVIDER", "sarvam")  # "sarvam" | "deepgram"
TTS_PROVIDER = os.getenv("VOICE_TTS_PROVIDER", "sarvam")  # "sarvam" | "cartesia"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
# REST /v1/listen (one-shot transcription) and the live /v2/listen streaming
# API use ENTIRELY DIFFERENT model families — v1 takes "nova-*", v2 only
# accepts "flux-general-en"/"flux-general-multi" (confirmed live: nova-3 on
# v2 gets rejected with a bare HTTP 400 "Unexpected error when initializing
# websocket connection", no further detail, right at the WS handshake).
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_STREAM_MODEL = os.getenv("DEEPGRAM_STREAM_MODEL", "flux-general-en")

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2024-06-10")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-2")
# Default voice — a stock Cartesia voice id. Overridable per-request later
# (e.g. a user-selectable voice), env default just for this build/test.
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "694f9389-aac1-45b6-b726-9d9369183238")
CARTESIA_SAMPLE_RATE = int(os.getenv("CARTESIA_SAMPLE_RATE", "24000"))

CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_REST_URL = "https://api.cartesia.ai/tts/bytes"
DEEPGRAM_REST_URL = "https://api.deepgram.com/v1/listen"

# ── Sarvam ───────────────────────────────────────────────────────────────
# Every field below was pulled from the ACTUAL sarvamai SDK source
# (raw_client.py / socket_client.py / types/*.py) and confirmed against the
# real API, not guessed from docs — docs describe an SDK-wrapped call shape,
# not the literal wire protocol, and the docs example speaker ("anushka")
# turned out to not even be valid for bulbul:v3 (confirmed live: 422 from
# the real API, which also usefully listed the actual valid v3 roster).
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
# Multiple prepaid keys can be given as one comma-separated value in this
# same env var — nothing else (GitHub secret name, merge_env.sh, deploy.yml)
# needs to know there's more than one. Whitespace around each key is
# stripped so "k1, k2, k3" and "k1,k2,k3" both work.
SARVAM_API_KEYS = [k.strip() for k in SARVAM_API_KEY.split(",") if k.strip()]


class _SarvamKeyPool:
    """Rotates through SARVAM_API_KEYS when the current one is rejected or
    reports it's out of balance, so a burned-through free-tier key doesn't
    take the voice pipeline down until someone notices and swaps it by hand.

    Deliberately in-memory, not persisted (Redis or otherwise) — this
    process is single-instance already (see turn_manager.py, llm_registry.py
    for the same "single-process, in-memory" precedent elsewhere in this
    codebase), and the cost of NOT persisting is only ever one wasted
    handshake against an already-dead key right after a redeploy, not a
    correctness problem. A lock isn't needed either: mark_exhausted() only
    ever advances the index forward and is a no-op if another concurrent
    caller already moved past the key being reported, so a race just means
    two callers independently agree to skip the same key.
    """

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._i = 0

    def __len__(self) -> int:
        return len(self._keys)

    def current(self) -> str:
        return self._keys[self._i] if self._keys else ""

    def mark_exhausted(self, key: str) -> None:
        if self._keys and self._i < len(self._keys) and self._keys[self._i] == key and self._i < len(self._keys) - 1:
            self._i += 1
            logger.warning(
                "voice.sarvam_key_pool.rotated",
                remaining_keys=len(self._keys) - self._i,
            )


SARVAM_KEY_POOL = _SarvamKeyPool(SARVAM_API_KEYS)

SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"
SARVAM_STT_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"

SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
# One of the roster returned by a live 422 against bulbul:v3 (anushka/
# manisha/etc are bulbul:v2-only despite being the docs' example default).
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")
SARVAM_TTS_LANGUAGE_CODE = os.getenv("SARVAM_TTS_LANGUAGE_CODE", "hi-IN")
SARVAM_TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "24000"))

# ── Audio wire format (browser-bound) ────────────────────────────────────
# The single format every TTS provider is normalized to before the audio
# goes out over the WebSocket, and the format voiceClient.js decodes.
#
# This is 8-bit mu-law, NOT the pcm_f32le this pipeline originally used, and
# the reason is bandwidth — measured, not guessed:
#
#   pcm_f32le @24kHz = 768 kbps   <- what we used to send
#   pcm_s16le @24kHz = 384 kbps
#   mulaw     @24kHz = 192 kbps   <- what we send now
#
# Measured transfer throughput to a real browser over the public HTTPS
# endpoint was ~414 kbps, so the old float32 stream needed roughly TWICE
# the bandwidth the link could actually carry. Audio therefore arrived at
# ~0.25x realtime and the player starved continuously — confirmed by
# instrumenting real turns (11.36s of speech taking 44.56s to deliver)
# while TTS itself measured 1.4-2.1x realtime and the agent produced its
# whole response in one 3.5s burst. Neither stage was slow; the pipe was.
#
# mu-law was chosen over mp3/opus (which Sarvam also offers, and which are
# far smaller still) because it is a stateless, sample-by-sample encoding:
# every chunk decodes independently with no frame boundaries, no container,
# no decoder library, and no change to the chunk-by-chunk streaming model.
# A compressed codec would cut bandwidth much further but would need
# MediaSource/decodeAudioData framing work to stay gapless.
VOICE_WIRE_CODEC = "mulaw"

SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
# codemix = "English words in English script and Indic words in native
# script" — confirmed live via a real TTS->STT round trip on a Hinglish
# sentence: 100% of the meaning survived (the model transliterated the
# English loanwords into Devanagari rather than keeping Latin script, which
# is a legitimate natural rendering of spoken Hindi, not an error).
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "codemix")
SARVAM_STT_LANGUAGE_CODE = os.getenv("SARVAM_STT_LANGUAGE_CODE", "hi-IN")
# The STT WS connection-level sample_rate param ONLY accepts 8000/16000
# (confirmed live: 24000 gets an immediate close with code 4000
# "Unsupported sample rate: 24000. Supported rates: 8000, 16000") — unlike
# the TTS side, which does accept 24000. Matches voice/stt.py's existing
# 16kHz mic-capture default, so no resampling is needed either way.
SARVAM_STT_SAMPLE_RATE = int(os.getenv("SARVAM_STT_SAMPLE_RATE", "16000"))


def require_keys() -> None:
    """Fail loudly at pipeline construction time, not mid-turn, if a
    required provider key is missing — matches this codebase's fail-fast
    posture for external-service credentials (see
    rag/tools/retrieval_tool.py's tenancy guard for the same philosophy
    applied to a different kind of missing required value). Only checks
    keys for the providers actually selected, so e.g. running Sarvam-only
    doesn't demand a Deepgram key that will never be used."""
    checks = []
    if STT_PROVIDER == "sarvam" or TTS_PROVIDER == "sarvam":
        checks.append(("SARVAM_API_KEY", SARVAM_API_KEY))
    if STT_PROVIDER == "deepgram":
        checks.append(("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY))
    if TTS_PROVIDER == "cartesia":
        checks.append(("CARTESIA_API_KEY", CARTESIA_API_KEY))
    missing = [name for name, val in checks if not val]
    if missing:
        raise RuntimeError(f"voice pipeline: missing required env vars: {', '.join(missing)}")


def require_stt_key() -> None:
    """Narrower than require_keys() — for STT-only callers (dictation:
    routes/voice_routes.py's /dictate endpoint), which have no TTS leg at
    all. Using the full require_keys() there would wrongly demand a TTS
    provider key (e.g. CARTESIA_API_KEY) that dictation will never touch,
    if STT_PROVIDER and TTS_PROVIDER ever point at different providers."""
    if STT_PROVIDER == "sarvam" and not SARVAM_API_KEY:
        raise RuntimeError("voice pipeline: missing required env var: SARVAM_API_KEY")
    if STT_PROVIDER == "deepgram" and not DEEPGRAM_API_KEY:
        raise RuntimeError("voice pipeline: missing required env var: DEEPGRAM_API_KEY")
