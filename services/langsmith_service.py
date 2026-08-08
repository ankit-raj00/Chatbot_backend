"""
LangSmithService — attaches real, backend-computed per-turn cost to the
corresponding LangSmith trace as feedback.

Tracing itself stays fully automatic (LANGCHAIN_TRACING_V2 env var, no manual
SDK calls anywhere else in this codebase) — this is the one place that talks
to the LangSmith SDK directly, and only to attach cost onto a run that's
already being traced. Needed because OmniRoute reports $0 for every call
through the "antigravity" provider pool, so LangSmith's own cost display is
useless for this model; the real number is computed in chat_service.py from
actual token counts × config/model_config.py's price table.

chat_service.py passes run_id=uuid.UUID(turn_id) into the graph invocation's
RunnableConfig, so the turn's already-minted turn_id IS the LangSmith root
run's ID — no second id to track.

IMPORTANT — verified live, not assumed: this uses Client.create_feedback(),
NOT Client.update_run(). update_run() was the original design and looked
correct from the SDK signature alone, but a live test against a real run
proved it fails in exactly the scenario this feature needs: LangGraph's own
tracer already sends a completion update when the graph finishes (that's
what makes the run show up as "done" in the UI at all), and LangSmith
rejects a second update_run() on an already-completed run with
"409 Conflict: Run update payload already received. Duplicate run update
requests for the same run are not supported." Every real production
attempt to attach cost via update_run() would hit this and silently fail
(swallowed by the try/except below) — the run would just never show cost.
create_feedback() is what LangSmith actually provides for exactly this
case — a separate, post-hoc annotation linked to a run_id that doesn't
touch the run record itself, so it can't conflict with the tracer's own
lifecycle. Confirmed live: shows up on read-back immediately (no
batching/ingestion delay, unlike the run record itself).
"""
import asyncio

import structlog
logger = structlog.get_logger(__name__)

_client = None  # lazily constructed, reused — mirrors graph/llm_registry.py's client-caching convention

ATTACH_RETRIES = 3
ATTACH_RETRY_DELAY_S = 2.0  # the run record itself can lag LangSmith's ingestion
                            # (its own tracer update is async/batched); the run must
                            # exist server-side before feedback can attach to it.


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
                    client.create_feedback,
                    run_id=run_id,
                    key="cost_usd",
                    value=round(cost_usd, 8),
                    extra={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "model": model,
                    },
                )
                return
            except Exception as e:
                if attempt == ATTACH_RETRIES:
                    logger.warning(f"Failed to attach cost to LangSmith run {run_id} after {ATTACH_RETRIES} attempts: {e}")
                else:
                    await asyncio.sleep(ATTACH_RETRY_DELAY_S)
