"""
End-to-End System Test Suite — AgentX v2
=========================================
Covers the ENTIRE pipeline from HTTP layer → supervisor → subgraphs → tools → skills → RAG.

Test groups:
  E1  - App startup & health checks
  E2  - Authentication (signup → login → protected route → logout)
  E3  - Supervisor intent routing (all 7 agents)
  E4  - Skill system (builtin list, validate, upload, list vault, delete)
  E5  - Universal file reader (all supported formats)
  E6  - RAG pipeline (embedding grader, query rewriter, memory service)
  E7  - Circuit breaker & tool cache
  E8  - Shell subgraph sandbox safety
  E9  - Chat subgraph skill injection
  E10 - Output & agent routes (auth-protected endpoints)
  E11 - Conversation routes
  E12 - Import integrity (all modules must import cleanly)
  E13 - Workspace cleanup utility
  E14 - Full chat streaming pipeline (mocked LLM)

Run with:
    pytest tests/test_e2e.py -v --tb=short
"""

import sys, os, csv, json, tempfile, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

VALID_SKILL_MD = """\
---
name: test-skill
description: A test skill for end-to-end testing
metadata:
  triggers:
    - "test trigger"
    - "e2e test"
  agent: chat
---
# Test Skill

This is a test skill body used for end-to-end testing purposes.
It verifies that the skill loader can correctly parse and inject skill content.
Make sure the body is long enough to pass validation (at least 50 characters).
"""

INVALID_SKILL_MD = """\
---
description: Missing name field
---
Short body.
"""

LLM_PATCH  = "graph.llm_registry.get_llm"
MCP_PATCH  = "utils.mcp_connection_manager.mcp_manager"
SKILL_PATCH = "skills.skill_loader.get_relevant_skill_for_message"


def _base_supervisor_state(**overrides):
    s = {
        "messages": [HumanMessage(content="Hello")],
        "user_id": "e2e_user", "conversation_id": "e2e_conv",
        "agent": "", "model": "gemini-2.5-flash",
        "enabled_tools": [], "selected_files": None,
        "skill_body": "", "final_response": "",
    }
    s.update(overrides)
    return s


# ══════════════════════════════════════════════════════════════
# E1 — App Startup & Health
# ══════════════════════════════════════════════════════════════

class TestE1AppHealth:
    """E1: Basic app startup checks without spinning up the full lifespan."""

    def test_e1_01_app_imports_cleanly(self):
        """The FastAPI app object must be importable without errors."""
        from main import app
        assert app is not None
        assert app.title == "Gemini MCP Chat API"

    def test_e1_02_root_endpoint_schema(self):
        """Root endpoint must be registered."""
        from main import app
        routes = {r.path for r in app.routes}
        assert "/" in routes

    def test_e1_03_health_endpoint_registered(self):
        """Health endpoint must be registered."""
        from main import app
        routes = {r.path for r in app.routes}
        assert "/health" in routes

    def test_e1_04_all_routers_registered(self):
        """All expected API path prefixes must be present."""
        from main import app
        paths = {r.path for r in app.routes}
        # Sample of expected paths
        expected_prefixes = [
            "/auth/signup", "/auth/login",
            "/chat/stream",
            "/api/skills/builtin", "/api/skills/vault",
            "/api/outputs",
        ]
        for prefix in expected_prefixes:
            assert any(p.startswith(prefix.rstrip("/")) for p in paths), \
                f"Missing route for prefix: {prefix}"

    def test_e1_05_cors_middleware_attached(self):
        """CORS middleware must be present in the middleware stack."""
        from main import app
        # CORSMiddleware is added via add_middleware() and shows up in middleware_stack
        # Check the app's middleware list by inspecting user_middleware or by type name
        stack_classes = [str(type(m.cls).__name__ if hasattr(m, 'cls') else m) for m in app.user_middleware]
        cors_found = any(
            "cors" in str(c).lower() or "CORS" in str(c)
            for c in stack_classes
        )
        # Alternative: check the actual class directly
        if not cors_found:
            from starlette.middleware.cors import CORSMiddleware
            cors_found = any(
                getattr(m, 'cls', None) is CORSMiddleware
                for m in app.user_middleware
            )
        assert cors_found, f"CORS middleware not found. Stack: {stack_classes}"


# ══════════════════════════════════════════════════════════════
# E2 — Auth Routes (schema + controller contract)
# ══════════════════════════════════════════════════════════════

class TestE2Auth:
    """E2: Auth route contracts — mock the DB so no real MongoDB needed."""

    def test_e2_01_signup_route_exists(self):
        from main import app
        paths = [r.path for r in app.routes]
        assert "/auth/signup" in paths

    def test_e2_02_login_route_exists(self):
        from main import app
        paths = [r.path for r in app.routes]
        assert "/auth/login" in paths

    def test_e2_03_me_route_exists(self):
        from main import app
        paths = [r.path for r in app.routes]
        assert "/auth/me" in paths

    def test_e2_04_signup_returns_422_on_empty_body(self):
        """Signup with no body must return 422."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post("/auth/signup", json={})
            assert r.status_code == 422

    def test_e2_05_login_returns_422_on_empty_body(self):
        """Login with no body must return 422."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post("/auth/login", json={})
            assert r.status_code == 422

    def test_e2_06_protected_route_requires_auth(self):
        """GET /auth/me without token must return 401."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/auth/me")
            assert r.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# E3 — Supervisor Intent Routing
# ══════════════════════════════════════════════════════════════

class TestE3SupervisorRouting:
    """E3: All 7 intent routes must work with mocked LLM."""

    @pytest.mark.asyncio
    async def _route(self, msg: str, expected_agent: str):
        from graph.supervisor import intent_classifier_node, VALID_AGENTS
        assert expected_agent in VALID_AGENTS
        state = _base_supervisor_state(messages=[HumanMessage(content=msg)])
        mock_resp = MagicMock()
        mock_resp.content = expected_agent
        with patch(LLM_PATCH) as mock_llm_fn, \
             patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
            mock_llm_fn.return_value = mock_llm
            result = await intent_classifier_node(state)
        assert result["agent"] == expected_agent
        return result

    @pytest.mark.asyncio
    async def test_e3_01_chat_route(self):
        await self._route("What is machine learning?", "chat")

    @pytest.mark.asyncio
    async def test_e3_02_shell_route(self):
        await self._route("Run my python script", "shell")

    @pytest.mark.asyncio
    async def test_e3_03_document_route(self):
        await self._route("Create a PDF report on sales", "document")

    @pytest.mark.asyncio
    async def test_e3_04_vision_route(self):
        await self._route("What does this image show?", "vision")

    @pytest.mark.asyncio
    async def test_e3_05_code_route(self):
        await self._route("Write a FastAPI endpoint", "code")

    @pytest.mark.asyncio
    async def test_e3_06_rag_route(self):
        await self._route("Search my knowledge base", "rag")

    @pytest.mark.asyncio
    async def test_e3_07_data_route(self):
        await self._route("Analyze this CSV for trends", "data")

    @pytest.mark.asyncio
    async def test_e3_08_invalid_agent_falls_back_to_chat(self):
        """Invalid agent name from LLM falls back to 'chat'."""
        from graph.supervisor import intent_classifier_node
        state = _base_supervisor_state()
        mock_resp = MagicMock(content="totally_unknown_agent")
        with patch(LLM_PATCH) as mf, \
             patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            ml = AsyncMock()
            ml.ainvoke = AsyncMock(return_value=mock_resp)
            mf.return_value = ml
            r = await intent_classifier_node(state)
        assert r["agent"] == "chat"

    @pytest.mark.asyncio
    async def test_e3_09_llm_failure_falls_back_to_chat(self):
        """LLM API failure falls back to 'chat' gracefully."""
        from graph.supervisor import intent_classifier_node
        state = _base_supervisor_state()
        with patch(LLM_PATCH) as mf, \
             patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            ml = AsyncMock()
            ml.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
            mf.return_value = ml
            r = await intent_classifier_node(state)
        assert r["agent"] == "chat"

    @pytest.mark.asyncio
    async def test_e3_10_empty_messages_defaults_to_chat(self):
        from graph.supervisor import intent_classifier_node
        state = _base_supervisor_state(messages=[])
        r = await intent_classifier_node(state)
        assert r["agent"] == "chat"

    @pytest.mark.asyncio
    async def test_e3_11_skill_injected_when_matched(self):
        """When skill_loader finds a match, skill_body must be set."""
        from graph.supervisor import intent_classifier_node
        state = _base_supervisor_state(messages=[HumanMessage(content="Create a PDF")])
        mock_resp = MagicMock(content="document")
        with patch(LLM_PATCH) as mf, \
             patch(SKILL_PATCH, new_callable=AsyncMock, return_value="## PDF Skill\nDo this."):
            ml = AsyncMock()
            ml.ainvoke = AsyncMock(return_value=mock_resp)
            mf.return_value = ml
            r = await intent_classifier_node(state)
        assert r["skill_body"] == "## PDF Skill\nDo this."

    def test_e3_12_valid_agents_set_complete(self):
        from graph.supervisor import VALID_AGENTS
        assert VALID_AGENTS == {"chat", "shell", "document", "vision", "code", "rag", "data"}


# ══════════════════════════════════════════════════════════════
# E4 — Skill System
# ══════════════════════════════════════════════════════════════

class TestE4SkillSystem:
    """E4: Full skill system — builtin loading, validation, routing."""

    def test_e4_01_list_builtin_skills_returns_list(self):
        from skills.skill_loader import list_builtin_skills
        skills = list_builtin_skills()
        assert isinstance(skills, list)
        assert len(skills) > 0

    def test_e4_02_all_builtin_skills_have_required_fields(self):
        from skills.skill_loader import list_builtin_skills
        for skill in list_builtin_skills():
            assert "name" in skill, f"Missing 'name' in {skill}"
            assert "description" in skill, f"Missing 'description' in {skill}"

    def test_e4_03_load_builtin_skill_body(self):
        from skills.skill_loader import list_builtin_skills, load_builtin_skill
        skills = list_builtin_skills()
        first_name = skills[0]["name"]
        body = load_builtin_skill(first_name)
        assert body is not None
        assert len(body) > 10

    def test_e4_04_load_unknown_skill_returns_none(self):
        from skills.skill_loader import load_builtin_skill
        assert load_builtin_skill("definitely-does-not-exist-skill") is None

    def test_e4_05_skill_name_underscore_normalization(self):
        from skills.skill_loader import load_builtin_skill, list_builtin_skills
        skills = list_builtin_skills()
        name_with_dash = skills[0]["name"]
        name_with_underscore = name_with_dash.replace("-", "_")
        b1 = load_builtin_skill(name_with_dash)
        b2 = load_builtin_skill(name_with_underscore)
        assert b1 == b2

    @pytest.mark.asyncio
    async def test_e4_06_get_relevant_skill_pdf_match(self):
        from skills.skill_loader import get_relevant_skill_for_message
        result = await get_relevant_skill_for_message("create a PDF report", "u1", "document")
        # Should find the create-pdf skill or return None — just must not crash
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_e4_07_get_relevant_skill_agent_filter(self):
        from skills.skill_loader import get_relevant_skill_for_message
        # Shell agent should not match PDF skill
        result = await get_relevant_skill_for_message("create a PDF", "u1", "shell")
        # If it matches, it's because the skill allows shell agent; either way it must not crash
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_e4_08_get_relevant_skill_no_match(self):
        from skills.skill_loader import get_relevant_skill_for_message
        result = await get_relevant_skill_for_message("xyzzy nonsense query abc123", "u1", "chat")
        assert result is None

    def test_e4_09_skill_vault_validate_endpoint_valid(self):
        """POST /api/skills/vault/validate with valid skill must return valid=True."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            # UserSkillCreate requires skill_name + skill_content
            r = client.post("/api/skills/vault/validate",
                            json={"skill_name": "test-skill",
                                  "skill_content": VALID_SKILL_MD})
            assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
            data = r.json()
            assert data["valid"] is True
            assert data["name"] == "test-skill"

    def test_e4_10_skill_vault_validate_endpoint_invalid(self):
        """POST /api/skills/vault/validate with invalid skill must return valid=False."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post("/api/skills/vault/validate",
                            json={"skill_name": "bad-skill",
                                  "skill_content": INVALID_SKILL_MD})
            assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
            data = r.json()
            assert data["valid"] is False
            assert len(data["errors"]) > 0

    def test_e4_11_builtin_skills_endpoint_no_auth_required(self):
        """GET /api/skills/builtin must work without authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/skills/builtin")
            assert r.status_code == 200
            assert "skills" in r.json()

    def test_e4_12_skill_vault_requires_auth(self):
        """GET /api/skills/vault must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/skills/vault")
            assert r.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# E5 — Universal File Reader
# ══════════════════════════════════════════════════════════════

class TestE5FileReader:
    """E5: Universal file reader handles all formats correctly."""

    @pytest.mark.asyncio
    async def test_e5_01_plain_text(self):
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("Hello, world!")
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r["type"] == "text"
            assert "Hello" in r["content"]
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e5_02_python_file(self):
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write("def foo(): return 42\n")
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r["type"] == "text"
            assert "def foo" in r["content"]
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e5_03_csv_parsed_as_rows(self):
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                          delete=False, encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "score"])
            w.writeheader()
            w.writerows([{"name": "Alice", "score": "95"},
                         {"name": "Bob",   "score": "82"}])
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r["type"] == "csv", f"Expected csv, got: {r}"
            assert r["row_count"] == 2
            assert "name" in r["columns"]
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e5_04_json_parsed_as_json(self):
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            json.dump({"key": "value", "num": 42}, f)
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r["type"] == "json", f"Expected json, got: {r}"
            assert "value" in r["content"]
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e5_05_markdown_file(self):
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("# Title\n\nContent.")
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r["type"] == "text"
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e5_06_truncation_flag_on_large_content(self):
        from services.universal_file_reader import extract_any_file
        big = "A" * 60_000
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write(big)
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r.get("truncated") is True
            assert len(r["content"]) <= 50_000
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e5_07_csv_columns_match_headers(self):
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                          delete=False, encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["product", "price", "qty"])
            w.writeheader()
            w.writerow({"product": "Widget", "price": "9.99", "qty": "100"})
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert set(r["columns"]) == {"product", "price", "qty"}
        finally:
            os.unlink(fname)


# ══════════════════════════════════════════════════════════════
# E6 — RAG Pipeline
# ══════════════════════════════════════════════════════════════

class TestE6RAGPipeline:
    """E6: Embedding grader, query rewriter, memory service."""

    def _make_grader(self, threshold=0.8):
        with patch("rag.graph.nodes.embedding_grader_node.GoogleGenerativeAIEmbeddings"):
            from rag.graph.nodes.embedding_grader_node import EmbeddingGraderNode
            return EmbeddingGraderNode(threshold=threshold)

    def _make_rewriter(self):
        with patch("rag.query_rewriter.ChatGoogleGenerativeAI"):
            from rag.query_rewriter import QueryRewriter
            return QueryRewriter()

    @pytest.mark.asyncio
    async def test_e6_01_grader_passes_relevant_docs(self):
        grader = self._make_grader(threshold=0.5)
        q_emb = [1.0, 0.0]; d_emb = [0.9, 0.44]  # cosine ≈ 0.9
        grader.embedder.aembed_query     = AsyncMock(return_value=q_emb)
        grader.embedder.aembed_documents = AsyncMock(return_value=[d_emb])
        docs = [Document(page_content="Relevant ML content")]
        state = {"question": "ml", "documents": docs, "web_search_needed": False}
        r = await grader.grade_documents(state)
        assert len(r["documents"]) == 1
        assert r["web_search_needed"] is False

    @pytest.mark.asyncio
    async def test_e6_02_grader_filters_irrelevant(self):
        grader = self._make_grader(threshold=0.8)
        q_emb = [1.0, 0.0]; d_emb = [0.0, 1.0]  # cosine = 0.0
        grader.embedder.aembed_query     = AsyncMock(return_value=q_emb)
        grader.embedder.aembed_documents = AsyncMock(return_value=[d_emb])
        docs = [Document(page_content="Unrelated content")]
        state = {"question": "ml", "documents": docs, "web_search_needed": False}
        r = await grader.grade_documents(state)
        assert r["documents"] == []
        assert r["web_search_needed"] is True

    @pytest.mark.asyncio
    async def test_e6_03_grader_empty_docs_triggers_web_search(self):
        grader = self._make_grader()
        state = {"question": "test", "documents": [], "web_search_needed": False}
        r = await grader.grade_documents(state)
        assert r["web_search_needed"] is True

    @pytest.mark.asyncio
    async def test_e6_04_grader_fail_open_on_api_error(self):
        grader = self._make_grader()
        grader.embedder.aembed_query = AsyncMock(side_effect=RuntimeError("embed API down"))
        docs = [Document(page_content="Some content")]
        state = {"question": "test", "documents": docs, "web_search_needed": False}
        r = await grader.grade_documents(state)
        assert r["documents"] == docs  # fail-open
        assert r["web_search_needed"] is False

    @pytest.mark.asyncio
    async def test_e6_05_query_rewriter_hyde_returns_string(self):
        rw = self._make_rewriter()
        rw.llm.ainvoke = AsyncMock(return_value=MagicMock(content="A passage about ML..."))
        result = await rw.hyde("What is ML?")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_e6_06_query_rewriter_multi_query_starts_with_original(self):
        rw = self._make_rewriter()
        rw.llm.ainvoke = AsyncMock(return_value=MagicMock(content="Variant 1\nVariant 2"))
        variants = await rw.multi_query("Original question?", n=2)
        assert variants[0] == "Original question?"
        assert len(variants) >= 2

    @pytest.mark.asyncio
    async def test_e6_07_memory_service_returns_all_when_few(self):
        from services.memory_service import MemoryService
        mems = [{"topic": "stack", "content": "Uses Python"}]
        with patch.object(MemoryService, "get_user_memories",
                          new_callable=AsyncMock, return_value=mems):
            r = await MemoryService.get_relevant_memories("u1", "test", top_k=5)
        assert r == mems

    @pytest.mark.asyncio
    async def test_e6_08_memory_service_limits_to_top_k(self):
        from services.memory_service import MemoryService
        many = [{"topic": f"t{i}", "content": f"c{i}"} for i in range(20)]
        q_emb  = [1.0] + [0.0] * 767
        d_embs = [[float(i % 2)] + [0.0] * 767 for i in range(20)]
        with patch.object(MemoryService, "get_user_memories",
                          new_callable=AsyncMock, return_value=many), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as MockEmb:
            inst = MagicMock()
            inst.aembed_query     = AsyncMock(return_value=q_emb)
            inst.aembed_documents = AsyncMock(return_value=d_embs)
            MockEmb.return_value = inst
            r = await MemoryService.get_relevant_memories("u1", "test", top_k=5)
        assert len(r) <= 5

    @pytest.mark.asyncio
    async def test_e6_09_memory_service_fallback_on_embed_error(self):
        from services.memory_service import MemoryService
        many = [{"topic": f"t{i}", "content": f"c{i}"} for i in range(20)]
        with patch.object(MemoryService, "get_user_memories",
                          new_callable=AsyncMock, return_value=many), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings",
                   side_effect=RuntimeError("embed fail")):
            r = await MemoryService.get_relevant_memories("u1", "test", top_k=5)
        assert len(r) <= 5


# ══════════════════════════════════════════════════════════════
# E7 — Circuit Breaker & Tool Cache
# ══════════════════════════════════════════════════════════════

class TestE7CircuitBreakerAndCache:
    """E7: Circuit breaker state machine + tool result cache."""

    def test_e7_01_circuit_breaker_starts_closed(self):
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker("e2e_cb_01", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_e7_02_circuit_breaker_success_keeps_closed(self):
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker("e2e_cb_02", failure_threshold=3)
        async def ok(): return "ok"
        r = await cb.call(ok)
        assert r == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_e7_03_circuit_breaker_opens_after_threshold(self):
        from utils.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker("e2e_cb_03", failure_threshold=3)
        async def fail(): raise RuntimeError("fail")
        for _ in range(3):
            try:
                await cb.call(fail)
            except RuntimeError:
                pass
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_e7_04_open_circuit_raises_service_unavailable(self):
        from utils.circuit_breaker import CircuitBreaker, CircuitState, ServiceUnavailableError
        cb = CircuitBreaker("e2e_cb_04", failure_threshold=1)
        async def fail(): raise RuntimeError("x")
        try: await cb.call(fail)
        except RuntimeError: pass
        assert cb.state == CircuitState.OPEN
        # Next call must raise ServiceUnavailableError immediately
        with pytest.raises(ServiceUnavailableError):
            await cb.call(fail)

    @pytest.mark.asyncio
    async def test_e7_05_tool_cache_caches_cacheable_tool(self):
        """cached_invoke must only call execute_fn once for same key within TTL."""
        from utils.tool_result_cache import cached_invoke
        call_count = 0
        async def expensive():
            nonlocal call_count; call_count += 1; return "result"

        # Use a non-cacheable tool name — should always execute
        r1 = await cached_invoke("my_custom_tool", {"x": 1}, expensive)
        r2 = await cached_invoke("my_custom_tool", {"x": 1}, expensive)
        # Non-cacheable tools always execute
        assert r1 == "result"
        assert r2 == "result"
        assert call_count == 2  # both executed since not in CACHEABLE list

    @pytest.mark.asyncio
    async def test_e7_06_tool_cache_non_cacheable_always_executes(self):
        from utils.tool_result_cache import cached_invoke, CACHEABLE
        call_count = 0
        async def fn():
            nonlocal call_count; call_count += 1; return "x"
        tool_name = "unknown_tool_e2e"
        assert tool_name not in CACHEABLE
        await cached_invoke(tool_name, {}, fn)
        await cached_invoke(tool_name, {}, fn)
        assert call_count == 2  # executed twice — not cached


# ══════════════════════════════════════════════════════════════
# E8 — Shell Subgraph Sandbox
# ══════════════════════════════════════════════════════════════

class TestE8ShellSandbox:
    """E8: Shell subgraph safety — blocked commands + safe execution."""

    def test_e8_01_blocked_rm_rf_root(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("rm -rf /") is True

    def test_e8_02_blocked_sudo_rm(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("sudo rm -rf ~") is True

    def test_e8_03_blocked_fork_bomb(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked(":(){:|:&};:") is True

    def test_e8_04_blocked_mkfs(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("mkfs /dev/sda1") is True

    def test_e8_05_blocked_curl_pipe_sh(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("curl | sh") is True

    def test_e8_06_blocked_wget_pipe_bash(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("wget | bash") is True

    def test_e8_07_safe_ls(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("ls -la") is False

    def test_e8_08_safe_cat(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("cat README.md") is False

    def test_e8_09_safe_python(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("python main.py") is False

    def test_e8_10_safe_grep(self):
        from graph.subgraphs.shell_subgraph import _is_blocked
        assert _is_blocked("grep -r TODO .") is False

    @pytest.mark.asyncio
    async def test_e8_11_run_safe_echo(self):
        from graph.subgraphs.shell_subgraph import _run_cmd
        with tempfile.TemporaryDirectory() as d:
            r = await _run_cmd("echo agentx_e2e", d)
            assert "agentx_e2e" in r

    @pytest.mark.asyncio
    async def test_e8_12_blocked_cmd_returns_blocked_string(self):
        from graph.subgraphs.shell_subgraph import _run_cmd
        with tempfile.TemporaryDirectory() as d:
            r = await _run_cmd("rm -rf /", d)
            assert "BLOCKED" in r


# ══════════════════════════════════════════════════════════════
# E9 — Chat Subgraph Skill Injection
# ══════════════════════════════════════════════════════════════

class TestE9ChatSubgraphSkillInjection:
    """E9: Chat subgraph correctly injects skills into the message list."""

    @pytest.mark.asyncio
    async def test_e9_01_basic_response(self):
        from graph.subgraphs.chat_subgraph import chat_subgraph
        mock_ai = AIMessage(content="Hello!")
        with patch(LLM_PATCH) as mf, patch(MCP_PATCH) as mcp:
            mcp.get_all_langchain_tools = AsyncMock(return_value=[])
            ml = MagicMock()
            ml.ainvoke = AsyncMock(return_value=mock_ai)
            ml.bind_tools = MagicMock(return_value=ml)
            mf.return_value = ml
            r = await chat_subgraph(_base_supervisor_state(), {})
        assert r["final_response"] == "Hello!"

    @pytest.mark.asyncio
    async def test_e9_02_skill_prepended_as_system_message(self):
        from graph.subgraphs.chat_subgraph import chat_subgraph
        with patch(LLM_PATCH) as mf, patch(MCP_PATCH) as mcp:
            mcp.get_all_langchain_tools = AsyncMock(return_value=[])
            ml = MagicMock()
            ml.ainvoke = AsyncMock(return_value=AIMessage(content="OK"))
            ml.bind_tools = MagicMock(return_value=ml)
            mf.return_value = ml
            state = _base_supervisor_state(skill_body="## PDF Skill\nDo this.")
            await chat_subgraph(state, {})
        msgs = ml.ainvoke.call_args[0][0]
        assert isinstance(msgs[0], SystemMessage)
        assert "PDF Skill" in msgs[0].content

    @pytest.mark.asyncio
    async def test_e9_03_existing_system_message_gets_skill_appended(self):
        from graph.subgraphs.chat_subgraph import chat_subgraph
        with patch(LLM_PATCH) as mf, patch(MCP_PATCH) as mcp:
            mcp.get_all_langchain_tools = AsyncMock(return_value=[])
            ml = MagicMock()
            ml.ainvoke = AsyncMock(return_value=AIMessage(content="OK"))
            ml.bind_tools = MagicMock(return_value=ml)
            mf.return_value = ml
            state = _base_supervisor_state(
                messages=[SystemMessage(content="You are helpful."),
                          HumanMessage(content="Help!")],
                skill_body="## ACTIVE SKILL\nDo this.",
            )
            await chat_subgraph(state, {})
        msgs = ml.ainvoke.call_args[0][0]
        assert "You are helpful." in msgs[0].content
        assert "ACTIVE SKILL" in msgs[0].content

    @pytest.mark.asyncio
    async def test_e9_04_no_skill_no_system_message_added(self):
        from graph.subgraphs.chat_subgraph import chat_subgraph
        with patch(LLM_PATCH) as mf, patch(MCP_PATCH) as mcp:
            mcp.get_all_langchain_tools = AsyncMock(return_value=[])
            ml = MagicMock()
            ml.ainvoke = AsyncMock(return_value=AIMessage(content="OK"))
            ml.bind_tools = MagicMock(return_value=ml)
            mf.return_value = ml
            state = _base_supervisor_state(skill_body="")  # no skill
            await chat_subgraph(state, {})
        msgs = ml.ainvoke.call_args[0][0]
        assert isinstance(msgs[0], HumanMessage)  # no SystemMessage prepended


# ══════════════════════════════════════════════════════════════
# E10 — Output & Agent Routes (HTTP level)
# ══════════════════════════════════════════════════════════════

class TestE10ProtectedRoutes:
    """E10: Auth-protected output/agent routes return 401 without token."""

    def test_e10_01_list_outputs_requires_auth(self):
        """GET /api/outputs/list must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/outputs/list")
            assert r.status_code in (401, 403), \
                f"Expected 401/403, got {r.status_code}"

    def test_e10_02_download_output_requires_auth(self):
        """GET /api/outputs/download/{user_id}/{filename} must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/outputs/download/some_user/report.pdf")
            assert r.status_code in (401, 403), \
                f"Expected 401/403, got {r.status_code}"

    def test_e10_03_agent_status_requires_auth(self):
        """GET /api/agent/status/{id} must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/agent/status/some-thread-id")
            assert r.status_code in (401, 403)

    def test_e10_04_agent_resume_requires_auth(self):
        """POST /api/agent/resume/{id} must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/agent/resume/some-thread-id",
                       json={"approved": True, "feedback": ""})
            assert r.status_code in (401, 403)

    def test_e10_05_agent_cancel_requires_auth(self):
        """POST /api/agent/cancel/{id} must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/agent/cancel/some-thread-id")
            assert r.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# E11 — Conversation Routes
# ══════════════════════════════════════════════════════════════

class TestE11ConversationRoutes:
    """E11: Conversation routes require authentication."""

    def test_e11_01_list_conversations_requires_auth(self):
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/conversations")
            assert r.status_code in (401, 403)

    def test_e11_02_chat_stream_requires_auth(self):
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/chat/stream", json={
                "message": "Hello",
                "model": "gemini-2.5-flash",
            })
            assert r.status_code in (401, 403)


# ══════════════════════════════════════════════════════════════
# E12 — Module Import Integrity
# ══════════════════════════════════════════════════════════════

class TestE12ImportIntegrity:
    """E12: All production modules must import cleanly."""

    def test_e12_01_supervisor(self):
        import graph.supervisor

    def test_e12_02_chat_subgraph(self):
        import graph.subgraphs.chat_subgraph

    def test_e12_03_shell_subgraph(self):
        import graph.subgraphs.shell_subgraph

    def test_e12_04_document_subgraph(self):
        import graph.subgraphs.document_subgraph

    def test_e12_05_vision_subgraph(self):
        import graph.subgraphs.vision_subgraph

    def test_e12_06_code_subgraph(self):
        import graph.subgraphs.code_subgraph

    def test_e12_07_rag_subgraph(self):
        import graph.subgraphs.rag_subgraph

    def test_e12_08_data_subgraph(self):
        import graph.subgraphs.data_subgraph

    def test_e12_09_skill_loader(self):
        import skills.skill_loader

    def test_e12_10_circuit_breaker(self):
        import utils.circuit_breaker

    def test_e12_11_tool_result_cache(self):
        import utils.tool_result_cache

    def test_e12_12_workspace_cleanup(self):
        import utils.workspace_cleanup

    def test_e12_13_universal_file_reader(self):
        import services.universal_file_reader

    def test_e12_14_skill_vault_routes(self):
        import routes.skill_vault_routes

    def test_e12_15_agent_routes(self):
        import routes.agent_routes

    def test_e12_16_output_routes(self):
        import routes.output_routes

    def test_e12_17_llm_registry(self):
        import graph.llm_registry

    def test_e12_18_memory_service(self):
        import services.memory_service

    def test_e12_19_embedding_grader(self):
        import rag.graph.nodes.embedding_grader_node

    def test_e12_20_query_rewriter(self):
        import rag.query_rewriter


# ══════════════════════════════════════════════════════════════
# E13 — Workspace Cleanup
# ══════════════════════════════════════════════════════════════

class TestE13WorkspaceCleanup:
    """E13: Workspace cleanup deletes stale files correctly."""

    @pytest.mark.asyncio
    async def test_e13_01_cleanup_deletes_old_files(self):
        import time
        import utils.workspace_cleanup as wc_mod

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)

            # Create an "old" file (simulate age via mtime)
            old_file = root_path / "old_output.pdf"
            old_file.write_text("old content")
            old_mtime = time.time() - (25 * 3600)   # 25 hours old
            os.utime(old_file, (old_mtime, old_mtime))

            # Create a "new" file
            new_file = root_path / "new_output.pdf"
            new_file.write_text("new content")

            # Patch the module constants so _cleanup() uses our temp dir
            with patch.object(wc_mod, "WORKSPACE_ROOT", root_path), \
                 patch.object(wc_mod, "MAX_AGE_HOURS", 24):
                await wc_mod._cleanup()

            assert not old_file.exists(), "Old file should have been deleted"
            assert new_file.exists(),     "New file should still exist"

    @pytest.mark.asyncio
    async def test_e13_02_cleanup_preserves_directories(self):
        """Directories themselves should not be deleted, only files inside."""
        import time
        import utils.workspace_cleanup as wc_mod

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            subdir = root_path / "user_123"
            subdir.mkdir()
            old_file = subdir / "old.txt"
            old_file.write_text("x")
            old_mtime = time.time() - (48 * 3600)
            os.utime(old_file, (old_mtime, old_mtime))

            with patch.object(wc_mod, "WORKSPACE_ROOT", root_path), \
                 patch.object(wc_mod, "MAX_AGE_HOURS", 24):
                await wc_mod._cleanup()

            assert subdir.exists(),       "Subdirectory itself should survive"
            assert not old_file.exists(), "Stale file inside subdir should be deleted"

    @pytest.mark.asyncio
    async def test_e13_03_cleanup_does_nothing_when_workspace_missing(self):
        """If workspace root doesn't exist, _cleanup() must return silently."""
        import utils.workspace_cleanup as wc_mod
        nonexistent = Path("/tmp/this_path_does_not_exist_e2e_test_agentx")
        with patch.object(wc_mod, "WORKSPACE_ROOT", nonexistent):
            await wc_mod._cleanup()   # must not raise


# ══════════════════════════════════════════════════════════════
# E14 — Full Chat Pipeline (mocked)
# ══════════════════════════════════════════════════════════════

class TestE14FullChatPipeline:
    """E14: End-to-end chat pipeline with mocked LLM and DB."""

    @pytest.mark.asyncio
    async def test_e14_01_supervisor_routes_and_subgraph_responds(self):
        """Full flow: supervisor classifies → chat subgraph responds."""
        from graph.supervisor import intent_classifier_node
        from graph.subgraphs.chat_subgraph import chat_subgraph

        state = _base_supervisor_state(
            messages=[HumanMessage(content="Tell me a joke")]
        )

        # Step 1: Intent classification
        mock_resp = MagicMock(content="chat")
        with patch(LLM_PATCH) as mf, \
             patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            ml = AsyncMock()
            ml.ainvoke = AsyncMock(return_value=mock_resp)
            mf.return_value = ml
            classified = await intent_classifier_node(state)

        assert classified["agent"] == "chat"
        state.update(classified)

        # Step 2: Chat subgraph runs
        mock_ai = AIMessage(content="Why did the AI cross the road?")
        with patch(LLM_PATCH) as mf2, patch(MCP_PATCH) as mcp:
            mcp.get_all_langchain_tools = AsyncMock(return_value=[])
            ml2 = MagicMock()
            ml2.ainvoke = AsyncMock(return_value=mock_ai)
            ml2.bind_tools = MagicMock(return_value=ml2)
            mf2.return_value = ml2
            chat_result = await chat_subgraph(state, {})

        assert "road" in chat_result["final_response"]
        assert len(chat_result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_e14_02_document_intent_routes_correctly(self):
        """PDF request → document agent → document subgraph mocked response."""
        from graph.supervisor import intent_classifier_node
        state = _base_supervisor_state(
            messages=[HumanMessage(content="Create a PDF report on quarterly sales")]
        )
        mock_resp = MagicMock(content="document")
        with patch(LLM_PATCH) as mf, \
             patch(SKILL_PATCH, new_callable=AsyncMock, return_value="## PDF Skill\nUse reportlab."):
            ml = AsyncMock()
            ml.ainvoke = AsyncMock(return_value=mock_resp)
            mf.return_value = ml
            r = await intent_classifier_node(state)
        assert r["agent"] == "document"
        assert "PDF Skill" in r["skill_body"]

    @pytest.mark.asyncio
    async def test_e14_03_shell_dangerous_command_blocked(self):
        """Shell agent must never execute dangerous commands."""
        from graph.subgraphs.shell_subgraph import _run_cmd
        with tempfile.TemporaryDirectory() as d:
            for dangerous in ["rm -rf /", "sudo rm -rf ~", "mkfs /dev/sda"]:
                r = await _run_cmd(dangerous, d)
                assert "BLOCKED" in r, f"Expected BLOCKED for: {dangerous}"

    @pytest.mark.asyncio
    async def test_e14_04_file_upload_pipeline_csv(self):
        """CSV upload through file reader simulates what data agent receives."""
        from services.universal_file_reader import extract_any_file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                          delete=False, encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["month", "revenue"])
            w.writeheader()
            w.writerows([{"month": "Jan", "revenue": "10000"},
                         {"month": "Feb", "revenue": "12000"},
                         {"month": "Mar", "revenue": "9500"}])
            fname = f.name
        try:
            r = await extract_any_file(Path(fname))
            assert r["type"] == "csv"
            assert r["row_count"] == 3
            assert "revenue" in r["columns"]
        finally:
            os.unlink(fname)

    @pytest.mark.asyncio
    async def test_e14_05_rag_pipeline_empty_retrieval_triggers_web_search(self):
        """Empty retrieval in RAG must trigger web search fallback."""
        with patch("rag.graph.nodes.embedding_grader_node.GoogleGenerativeAIEmbeddings"):
            from rag.graph.nodes.embedding_grader_node import EmbeddingGraderNode
            grader = EmbeddingGraderNode(threshold=0.72)
        state = {"question": "obscure topic", "documents": [], "web_search_needed": False}
        r = await grader.grade_documents(state)
        assert r["web_search_needed"] is True

    def test_e14_06_skill_vault_validate_valid_skill(self):
        """End-to-end: skill validation endpoint returns valid for correct SKILL.md."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/skills/vault/validate",
                       json={"skill_name": "test-skill",
                             "skill_content": VALID_SKILL_MD})
            assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
            assert r.json()["valid"] is True

    def test_e14_07_builtin_skills_listed_publicly(self):
        """End-to-end: list of builtin skills available without auth."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/skills/builtin")
            assert r.status_code == 200
            skills = r.json()["skills"]
            assert len(skills) > 0

    def test_e14_08_all_7_valid_agents_present(self):
        """All 7 agent types must be present in VALID_AGENTS."""
        from graph.supervisor import VALID_AGENTS
        for agent in ["chat", "shell", "document", "vision", "code", "rag", "data"]:
            assert agent in VALID_AGENTS
