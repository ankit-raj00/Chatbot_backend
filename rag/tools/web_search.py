"""
TinyFish-backed web search for the RAG pipeline (replaces Tavily).

Two-step for accuracy:
  1. search.query(...)                    → ranked candidates (title/url/snippet)
  2. fetch.get_contents([top urls], md)   → full page markdown for the top hits
     (one BATCHED call — the SDK accepts a list of urls)

Snippets are short; deep-fetching the top results' full markdown gives the
generator real grounding text. Deep fetch is bounded (WEB_SEARCH_DEEP_COUNT)
and can be disabled (WEB_SEARCH_DEEP_FETCH=false) to trade accuracy for speed.

The TinyFish SDK is synchronous and has no async client, so async callers must
go through `search_web()` which offloads to a thread.
"""
import os
import asyncio
from typing import List

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

_client = None


def _get_client():
    """Lazy singleton TinyFish client. Reads TINYFISH_API_KEY from env."""
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("TINYFISH_API_KEY")
    if not api_key:
        logger.warning("TINYFISH_API_KEY not set — web search disabled")
        return None
    try:
        from tinyfish import TinyFish
        _client = TinyFish(api_key=api_key)
    except Exception as e:
        logger.warning("TinyFish client init failed", error=str(e))
        _client = None
    return _client


def search_web_sync(query: str, max_results: int = 3) -> List[Document]:
    """
    Synchronous web search. Returns LangChain Documents:
        page_content = full markdown (deep-fetched) or snippet (fallback)
        metadata     = {"source": url, "title": ..., "type": "web"}

    Never raises — on any failure returns [] so a caller's turn still succeeds.
    """
    client = _get_client()
    if client is None or not query or not query.strip():
        return []

    deep = os.getenv("WEB_SEARCH_DEEP_FETCH", "true").lower() == "true"
    deep_count = int(os.getenv("WEB_SEARCH_DEEP_COUNT", "2"))

    try:
        resp = client.search.query(query, location="US", language="en")
        results = list(getattr(resp, "results", []) or [])[:max_results]
    except Exception as e:
        logger.warning("TinyFish search failed", error=str(e), query=query)
        return []

    if not results:
        return []

    # Map url -> full markdown for the top `deep_count` hits (single batched fetch)
    full_text: dict = {}
    if deep and deep_count > 0:
        top_urls = [r.url for r in results[:deep_count] if getattr(r, "url", None)]
        if top_urls:
            try:
                fetched = client.fetch.get_contents(top_urls, format="markdown")
                for page in getattr(fetched, "results", []) or []:
                    url = getattr(page, "final_url", None) or getattr(page, "url", None)
                    text = getattr(page, "text", None) or getattr(page, "content", None)
                    if url and text:
                        full_text[url] = text
            except Exception as e:
                logger.warning("TinyFish fetch failed (using snippets)", error=str(e))

    docs: List[Document] = []
    for r in results:
        url = getattr(r, "url", "") or "web_search"
        content = full_text.get(url) or getattr(r, "snippet", "") or ""
        if not content.strip():
            continue
        docs.append(Document(
            page_content=content,
            metadata={
                "source": url,
                "title": getattr(r, "title", "") or "",
                "type": "web",
            },
        ))
    logger.info("web search done", provider="tinyfish", results=len(docs), deep=deep)
    return docs


async def search_web(query: str, max_results: int = 3) -> List[Document]:
    """Async wrapper — the TinyFish SDK is sync, so run it in a thread."""
    return await asyncio.to_thread(search_web_sync, query, max_results)
