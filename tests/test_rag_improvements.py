"""Tests for RAG improvements — embedding grader and query rewriter."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.documents import Document


# ── EmbeddingGraderNode Tests ──────────────────────────────────────────────────

def _make_grader(threshold=0.8):
    """Construct a grader with a mocked embedder so no API calls are made."""
    with patch("rag.graph.nodes.embedding_grader_node.GoogleGenerativeAIEmbeddings"):
        from rag.graph.nodes.embedding_grader_node import EmbeddingGraderNode
        grader = EmbeddingGraderNode(threshold=threshold)
    return grader


@pytest.mark.asyncio
async def test_embedding_grader_filters_irrelevant():
    """Documents below threshold should be filtered out."""
    grader = _make_grader(threshold=0.8)

    q_emb            = [1.0, 0.0]
    d_emb_relevant   = [0.95, 0.31]   # cosine ≈ 0.95 — above 0.8
    d_emb_irrelevant = [0.0,  1.0]    # cosine = 0.0  — below 0.8

    docs = [
        Document(page_content="Relevant document about machine learning"),
        Document(page_content="Totally unrelated content about cooking"),
    ]

    grader.embedder.aembed_query     = AsyncMock(return_value=q_emb)
    grader.embedder.aembed_documents = AsyncMock(return_value=[d_emb_relevant, d_emb_irrelevant])

    state = {"question": "machine learning", "documents": docs, "web_search_needed": False}
    result = await grader.grade_documents(state)

    assert len(result["documents"]) == 1
    assert "machine learning" in result["documents"][0].page_content
    assert result["web_search_needed"] is False


@pytest.mark.asyncio
async def test_embedding_grader_all_irrelevant_triggers_web_search():
    """When all docs are below threshold, web_search_needed must become True."""
    grader = _make_grader(threshold=0.8)

    q_emb  = [1.0, 0.0]
    d_emb  = [0.0, 1.0]   # cosine = 0.0

    docs = [Document(page_content="Off-topic document")]
    grader.embedder.aembed_query     = AsyncMock(return_value=q_emb)
    grader.embedder.aembed_documents = AsyncMock(return_value=[d_emb])

    state = {"question": "test", "documents": docs, "web_search_needed": False}
    result = await grader.grade_documents(state)

    assert result["documents"] == []
    assert result["web_search_needed"] is True


@pytest.mark.asyncio
async def test_embedding_grader_empty_docs_triggers_web_search():
    """Empty docs list should set web_search_needed=True."""
    grader = _make_grader(threshold=0.72)
    state = {"question": "test", "documents": [], "web_search_needed": False}
    result = await grader.grade_documents(state)
    assert result["web_search_needed"] is True
    assert result["documents"] == []


@pytest.mark.asyncio
async def test_embedding_grader_fail_open_on_error():
    """Embedding API failure should fail-open (return all docs, no web search forced)."""
    grader = _make_grader(threshold=0.72)
    docs = [Document(page_content="Some content")]

    grader.embedder.aembed_query = AsyncMock(side_effect=RuntimeError("API down"))

    state = {"question": "test", "documents": docs, "web_search_needed": False}
    result = await grader.grade_documents(state)

    assert result["documents"] == docs
    assert result["web_search_needed"] is False


# ── QueryRewriter Tests ────────────────────────────────────────────────────────

def _make_rewriter():
    """Construct a QueryRewriter with a mocked LLM."""
    with patch("rag.query_rewriter.ChatGoogleGenerativeAI"):
        from rag.query_rewriter import QueryRewriter
        rw = QueryRewriter()
    return rw


@pytest.mark.asyncio
async def test_query_rewriter_hyde():
    """HyDE should return a non-empty passage."""
    rw = _make_rewriter()
    mock_resp = MagicMock()
    mock_resp.content = "Machine learning is a branch of AI that..."
    rw.llm.ainvoke = AsyncMock(return_value=mock_resp)

    result = await rw.hyde("What is machine learning?")
    assert "Machine learning" in result


@pytest.mark.asyncio
async def test_query_rewriter_multi_query_includes_original():
    """multi_query must always include the original question as first element."""
    rw = _make_rewriter()
    mock_resp = MagicMock()
    mock_resp.content = "How does ML work?\nWhat is AI learning?\nExplain supervised learning"
    rw.llm.ainvoke = AsyncMock(return_value=mock_resp)

    variants = await rw.multi_query("What is ML?", n=3)
    assert variants[0] == "What is ML?"   # original always first
    assert len(variants) >= 2


@pytest.mark.asyncio
async def test_query_rewriter_multi_query_strips_empty_lines():
    """Empty lines in LLM output should be stripped."""
    rw = _make_rewriter()
    mock_resp = MagicMock()
    mock_resp.content = "\nVariant 1\n\nVariant 2\n"
    rw.llm.ainvoke = AsyncMock(return_value=mock_resp)

    variants = await rw.multi_query("Original?", n=2)
    # Only non-empty strings should appear after the original
    for v in variants:
        assert v.strip() != ""


# ── MemoryService.get_relevant_memories Tests ──────────────────────────────────

@pytest.mark.asyncio
async def test_get_relevant_memories_returns_all_when_few():
    """When total memories ≤ top_k, return all without embedding calls."""
    from services.memory_service import MemoryService

    few_mems = [{"topic": "stack", "content": "Uses Python"} for _ in range(3)]
    with patch.object(MemoryService, "get_user_memories", new_callable=AsyncMock, return_value=few_mems):
        result = await MemoryService.get_relevant_memories("u1", "test", top_k=5)
    assert result == few_mems


@pytest.mark.asyncio
async def test_get_relevant_memories_returns_top_k():
    """get_relevant_memories must return at most top_k items when there are many memories."""
    from services.memory_service import MemoryService

    many_mems = [{"topic": f"t{i}", "content": f"content {i}"} for i in range(20)]

    # Build realistic embeddings
    q_emb  = [1.0] + [0.0] * 767
    d_embs = [[float(i % 2)] + [0.0] * 767 for i in range(20)]

    with patch.object(MemoryService, "get_user_memories", new_callable=AsyncMock, return_value=many_mems), \
         patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as MockEmb:

        instance = MagicMock()
        instance.aembed_query     = AsyncMock(return_value=q_emb)
        instance.aembed_documents = AsyncMock(return_value=d_embs)
        MockEmb.return_value = instance

        result = await MemoryService.get_relevant_memories("u1", "test message", top_k=5)

    assert len(result) <= 5


@pytest.mark.asyncio
async def test_get_relevant_memories_falls_back_on_embed_error():
    """On embedding failure, must fall back to top_k slice of all memories."""
    from services.memory_service import MemoryService

    many_mems = [{"topic": f"t{i}", "content": f"content {i}"} for i in range(20)]

    with patch.object(MemoryService, "get_user_memories", new_callable=AsyncMock, return_value=many_mems), \
         patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", side_effect=RuntimeError("embed err")):
        result = await MemoryService.get_relevant_memories("u1", "test", top_k=5)

    assert len(result) <= 5
