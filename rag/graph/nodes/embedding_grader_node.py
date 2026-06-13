"""
Embedding-based relevance grader.

Replaces the LLM-based GraderNode.

LLM grader:       1 API call, 500-2000ms, ~$0.00005/query
Embedding grader: 1 batched embed call, 100-300ms, ~$0.000001/query

The embedding approach is faster, cheaper, and more consistent.
"""

import asyncio
import os
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag.graph.state import RAGGraphState
import structlog

logger = structlog.get_logger(__name__)
THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.72"))


class EmbeddingGraderNode:
    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.embedder  = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            output_dimensionality=768,
            max_retries=3,
        )

    async def grade_documents(self, state: RAGGraphState) -> RAGGraphState:
        question  = state["question"]
        documents = state["documents"]

        if not documents:
            return {**state, "documents": [], "web_search_needed": True}

        try:
            q_emb, d_embs = await asyncio.gather(
                self.embedder.aembed_query(question),
                self.embedder.aembed_documents(
                    [d.page_content[:600] for d in documents]
                ),
            )
            q = np.array(q_emb)
            filtered = []
            for i, (doc, de) in enumerate(zip(documents, d_embs)):
                score = float(np.dot(q, np.array(de)))
                doc.metadata["relevance_score"] = score
                if score >= self.threshold:
                    logger.info(f"  ✅ [{i+1}] score={score:.3f}")
                    filtered.append(doc)
                else:
                    logger.info(f"  ❌ [{i+1}] score={score:.3f} filtered")

            return {
                **state,
                "documents": filtered,
                "web_search_needed": len(filtered) == 0,
            }
        except Exception as e:
            logger.warning(f"EmbeddingGrader failed ({e}) — fail-open")
            return {**state, "documents": documents, "web_search_needed": False}
