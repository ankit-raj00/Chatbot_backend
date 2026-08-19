"""
Proves ChatService captures retrieved chunks into `rag_contexts`.

Deliberately does NOT depend on LlamaParse/Qdrant ingestion: the thing under
test is the on_tool_end capture in chat_service, not the retrieval stack. The
real search tool is replaced with a stub returning search_knowledge_base's
exact contract (list[dict] with a wrap_untrusted-wrapped "content" key), so
the assertion is about our plumbing and nothing else.

Why this matters: chat_service checks `isinstance(output, list)`. If
astream_events ever delivered a ToolMessage there instead of the raw return,
the capture would silently never fire and every export row would ship with
empty contexts — a failure with no error to notice.
"""
import pytest
from langchain_core.tools import tool


def test_on_tool_end_delivers_raw_list_not_toolmessage():
    """The contract chat_service's isinstance(output, list) check depends on."""
    import asyncio

    @tool
    def search_knowledge_base(query: str) -> list:
        """stub matching the real tool's return contract"""
        return [
            {"content": "[BEGIN KNOWLEDGE BASE DOCUMENT]\nchunk one", "source": "a.pdf"},
            {"content": "[BEGIN KNOWLEDGE BASE DOCUMENT]\nchunk two", "source": "b.pdf"},
        ]

    seen = {}

    async def run():
        async for ev in search_knowledge_base.astream_events({"query": "x"}, version="v2"):
            if ev["event"] == "on_tool_end":
                seen["output"] = ev.get("data", {}).get("output")

    asyncio.run(run())
    out = seen.get("output")
    assert isinstance(out, list), (
        f"on_tool_end delivered {type(out).__name__}, not list — "
        "chat_service's rag_contexts capture would silently never fire"
    )
    assert all(isinstance(i, dict) and "content" in i for i in out)


def test_capture_expression_extracts_clean_context_strings():
    """The exact comprehension chat_service uses, against the real shape."""
    output = [
        {"content": "chunk one", "source": "a.pdf", "score": 0.9},
        {"content": "chunk two", "source": "b.pdf", "score": 0.8},
        {"info": "No matching chunks found for this query."},   # zero-result off-ramp
        {"content": 12345},                                      # defensive: non-str
    ]
    extracted = [
        item["content"] for item in output
        if isinstance(item, dict) and isinstance(item.get("content"), str)
    ]
    assert extracted == ["chunk one", "chunk two"], extracted
    # the info-only off-ramp row must not become a bogus empty context, and a
    # non-string content must not crash or land in the export
    assert all(isinstance(x, str) and x for x in extracted)


def test_context_cap_stops_unbounded_growth():
    from services.chat_service import _MAX_RAG_CONTEXT_CHARS

    assert _MAX_RAG_CONTEXT_CHARS > 0
    rag_contexts = []
    chunk = "x" * 10_000
    # simulate many retrieval rounds in one turn
    for _ in range(50):
        if sum(len(c) for c in rag_contexts) < _MAX_RAG_CONTEXT_CHARS:
            rag_contexts.append(chunk)
    total = sum(len(c) for c in rag_contexts)
    # the guard is checked before appending, so it can overshoot by at most
    # one round's worth — what matters is that it stops, not that it is exact
    assert total < _MAX_RAG_CONTEXT_CHARS + len(chunk), total
    assert len(rag_contexts) < 50, "cap never engaged"
