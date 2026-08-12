"""
AgentX voice pipeline — a self-contained module bridging real-time speech
and the existing text-based agent (graph/builder.py).

    audio in -> Deepgram STT -> existing LangGraph agent -> Cartesia TTS -> audio out

Kept separate from services/chat_service.py deliberately: voice has a
different transport (bidirectional audio, not SSE), a different system
prompt (spoken, not markdown-formatted), and different latency constraints
(must start speaking before the full response is ready). It reuses the same
agent graph so voice queries get the same tools/RAG as chat, not a
downgraded voice-only assistant.
"""
from voice.pipeline import VoicePipeline

__all__ = ["VoicePipeline"]
