"""
CPU cross-encoder reranker (FastEmbed / ONNX — no torch, no API).

Retrieval casts a wide net (top-N by vector similarity); the reranker then
scores each candidate against the query with a cross-encoder and keeps the best
top-K. This is the single biggest precision lever after embeddings, and it runs
locally on CPU with no external calls.
"""
import os
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)

# ms-marco MiniLM is ~5-10x faster than bge-reranker-base on CPU with strong
# quality — the right tradeoff for interactive chat. Override via env for higher
# accuracy (BAAI/bge-reranker-base) when latency is less critical.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

_reranker = None


def get_reranker():
    """Lazy singleton — the model loads once (first call downloads it)."""
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        logger.info("loading reranker", model=RERANKER_MODEL)
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    docs: List[Dict[str, Any]],
    text_key: str = "content",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Reorder `docs` by cross-encoder relevance to `query`, return the top_k.
    Each returned doc gets a `rerank_score`. Never raises — on any failure it
    falls back to the original order (so retrieval still works).
    """
    if not RERANK_ENABLED or not docs or len(docs) <= 1:
        return docs[:top_k]
    try:
        texts = [str(d.get(text_key) or "") for d in docs]
        scores = list(get_reranker().rerank(query, texts))
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        ranked = sorted(docs, key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]
    except Exception as e:  # never let reranking break retrieval
        logger.warning("rerank failed — using original order", error=str(e))
        return docs[:top_k]
