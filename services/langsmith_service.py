"""
LangSmithService — attaches real, backend-computed per-turn cost to the
corresponding LangSmith trace.

Tracing itself stays fully automatic (LANGCHAIN_TRACING_V2 env var, no manual
SDK calls anywhere else in this codebase) — this is the one place that talks
to the LangSmith SDK directly, and only to patch cost onto a run that's
already being traced. Needed because OmniRoute reports $0 for every call
through the "antigravity" provider pool, so LangSmith's own cost display is
useless for this model; the real number is computed in chat_service.py from
actual token counts × config/model_config.py's price table.

chat_service.py passes run_id=uuid.UUID(turn_id) into the graph invocation's
RunnableConfig, so the turn's already-minted turn_id IS the LangSmith root
run's ID — no second id to track.
"""
import asyncio
from typing import Optional

import structlog
logger = structlog.get_logger(__name__)

_client = None  # lazily constructed, reused — mirrors graph/llm_registry.py's client-caching convention

ATTACH_RETRIES = 3
ATTACH_RETRY_DELAY_S = 2.0  # LangSmith's run ingestion is async/batched — the
                            # run may not exist yet immediately after the
                            # turn finishes, so a couple short retries.


def _get_client():
    global _client
    if _client is None:
        from langsmith import Client
        _client = Client()
    return _client


class LangSmithService:

    @staticmethod
    async def attach_cost(
        run_id: str,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        """Best-effort, background-only — must never raise into the caller
        (chat_service.py's Step 8b spawns this and moves on)."""
        for attempt in range(1, ATTACH_RETRIES + 1):
            try:
                client = _get_client()
                await asyncio.to_thread(
                    client.update_run,
                    run_id,
                    extra={"metadata": {
                        "cost_usd": round(cost_usd, 8),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "model": model,
                    }},
                )
                return
            except Exception as e:
                if attempt == ATTACH_RETRIES:
                    logger.warning(f"Failed to attach cost to LangSmith run {run_id} after {ATTACH_RETRIES} attempts: {e}")
                else:
                    await asyncio.sleep(ATTACH_RETRY_DELAY_S)
