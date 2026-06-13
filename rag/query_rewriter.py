"""
Query rewriting — improves retrieval quality before vector search.

HyDE (Hypothetical Document Embedding):
  Generate a fake answer → embed it → use that embedding for retrieval.
  Why: The fake answer lives in the same embedding space as actual documents,
  matching terminology better than the raw question.

Multi-query:
  Generate N rephrasings → retrieve for each → merge + dedup.
  Why: Different phrasings capture different parts of the answer.
"""

import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
import structlog

logger = structlog.get_logger(__name__)


class QueryRewriter:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    async def hyde(self, question: str) -> str:
        """Generate a hypothetical document that would answer the question."""
        prompt = (
            f"Write a short, factual passage (2-4 sentences) that would be a "
            f"perfect answer to this question. Write it as document text.\n\n"
            f"Question: {question}\n\nPassage:"
        )
        r = await self.llm.ainvoke(prompt)
        return r.content.strip()

    async def multi_query(self, question: str, n: int = 3) -> List[str]:
        """Generate n alternative phrasings of the question."""
        prompt = (
            f"Generate {n} different ways to phrase this question. "
            f"One per line, no numbering.\n\nQuestion: {question}\n\nVariants:"
        )
        r = await self.llm.ainvoke(prompt)
        variants = [l.strip() for l in r.content.strip().split("\n") if l.strip()]
        return [question] + variants[:n]
