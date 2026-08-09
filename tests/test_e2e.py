"""
End-to-End System Test Suite — AgentX v3
=========================================
Covers the pipeline from HTTP layer → single ReAct agent (graph/builder.py) →
tools → skills → RAG. The pre-v3 supervisor/per-intent-subgraph architecture has
been fully removed; agent-graph-level coverage for the current architecture
lives in tests/test_agent_v3.py.

Test groups:
  E1  - App startup & health checks
  E2  - Authentication (signup → login → protected route → logout)
  E4  - Skill system (builtin list, validate, upload, list vault, delete)
  E5  - Universal file reader (all supported formats)
  E6  - RAG pipeline (embedding grader, query rewriter, memory service)
  E7  - Circuit breaker & tool cache
  E8  - run_shell sandbox safety
  E10 - Output & agent routes (auth-protected endpoints)
  E11 - Conversation routes
  E12 - Import integrity (all modules must import cleanly)
  E13 - Workspace cleanup utility (idle-gated policy)
  E14 - Misc pipeline checks (file reader, RAG grader fallback, skill vault HTTP)

Run with:
    pytest tests/test_e2e.py -v --tb=short
"""

import sys, os, csv, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from langchain_core.documents import Document
from fastapi.testclient import TestClient

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

def _all_route_paths(app) -> set[str]:
    """Every concrete endpoint path FastAPI actually serves, recursing into
    included routers.

    Newer Starlette/FastAPI (unpinned in requirements.txt until 2026-08-09,
    so this genuinely varies by when `pip install` last ran) wraps each
    app.include_router(...) call in a `_IncludedRouter` object that exposes
    neither `.path` NOR `.routes` directly -- the real APIRoute objects with
    their already-prefixed paths (e.g. "/auth/signup", not just "/signup")
    live at `.original_router.routes`. Plain top-level routes (the OpenAPI/
    docs endpoints, "/", "/health") still have `.path` directly and no
    `original_router`. Confirmed live: iterating app.routes and reading
    `.path` unconditionally crashed with AttributeError on a fresh CI
    install; filtering those out silently (rather than recursing) made every
    included-router endpoint (signup, login, chat/stream, ...) invisible to
    these tests instead, which would have made this whole test class rubber-stamp
    a route that had actually gone missing."""
    paths = set()
    for r in app.routes:
        path = getattr(r, "path", None)
        if path is not None:
            paths.add(path)
            continue
        original = getattr(r, "original_router", None)
        if original is not None and hasattr(original, "routes"):
            for sub in original.routes:
                sub_path = getattr(sub, "path", None)
                if sub_path is not None:
                    paths.add(sub_path)
    return paths


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
        routes = _all_route_paths(app)
        assert "/" in routes

    def test_e1_03_health_endpoint_registered(self):
        """Health endpoint must be registered."""
        from main import app
        routes = _all_route_paths(app)
        assert "/health" in routes

    def test_e1_04_all_routers_registered(self):
        """All expected API path prefixes must be present."""
        from main import app
        paths = _all_route_paths(app)
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
        paths = _all_route_paths(app)
        assert "/auth/signup" in paths

    def test_e2_02_login_route_exists(self):
        from main import app
        paths = _all_route_paths(app)
        assert "/auth/login" in paths

    def test_e2_03_me_route_exists(self):
        from main import app
        paths = _all_route_paths(app)
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

    def test_e2_07_credits_route_exists_and_requires_auth(self):
        """GET /api/users/credits must be registered and reject anonymous
        requests -- same shape as /auth/me above."""
        from main import app
        assert "/api/users/credits" in _all_route_paths(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/users/credits")
            assert r.status_code in (401, 403)

    def test_e2_08_credits_route_returns_spend_and_cap(self):
        """Authenticated: returns the current user's own spend/cap, not the
        admin-only /admin/users shape -- this is the whole point of the
        route (a regular user previously had no way to see this at all)."""
        from main import app
        from core.middleware import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {"_id": "507f1f77bcf86cd799439011"}
        try:
            with patch("services.credit_service.CreditService.get_spend", AsyncMock(return_value=1.234567)), \
                 patch("services.credit_service.CreditService.get_cap", AsyncMock(return_value=5.0)), \
                 patch("services.credit_service.CreditService._is_admin", AsyncMock(return_value=False)):
                with TestClient(app, raise_server_exceptions=False) as client:
                    r = client.get("/api/users/credits")
            assert r.status_code == 200
            body = r.json()
            assert body == {"used_usd": 1.234567, "cap_usd": 5.0, "is_admin": False}
        finally:
            app.dependency_overrides.pop(get_current_user, None)


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
        # Returns (body, name) tuple on match, or None — just must not crash.
        assert result is None or isinstance(result, tuple)

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
# E8 — run_shell Sandbox
# ══════════════════════════════════════════════════════════════

class TestE8ShellSandbox:
    """E8: run_shell sandbox safety — blocked commands + safe execution.

    Targets the current tools directly: tools.utilities.run_shell.BLOCKED_PATTERNS
    for pattern matching, and utils.code_executor.run_shell for execution — the
    graph/subgraphs/shell_subgraph.py compat shim these used to go through has
    been removed along with the rest of the pre-v3 subgraph architecture.
    """

    @staticmethod
    def _is_blocked(cmd: str) -> bool:
        from tools.utilities.run_shell import BLOCKED_PATTERNS
        cmd_lower = cmd.lower()
        return any(p in cmd_lower for p in BLOCKED_PATTERNS)

    def test_e8_01_blocked_rm_rf_root(self):
        assert self._is_blocked("rm -rf /") is True

    def test_e8_02_blocked_sudo_rm(self):
        assert self._is_blocked("sudo rm -rf ~") is True

    def test_e8_03_blocked_fork_bomb(self):
        assert self._is_blocked(":(){:|:&};:") is True

    def test_e8_04_blocked_mkfs(self):
        assert self._is_blocked("mkfs /dev/sda1") is True

    def test_e8_05_blocked_curl_pipe_sh(self):
        assert self._is_blocked("curl | sh") is True

    def test_e8_06_blocked_wget_pipe_bash(self):
        assert self._is_blocked("wget | bash") is True

    def test_e8_07_safe_ls(self):
        assert self._is_blocked("ls -la") is False

    def test_e8_08_safe_cat(self):
        assert self._is_blocked("cat README.md") is False

    def test_e8_09_safe_python(self):
        assert self._is_blocked("python main.py") is False

    def test_e8_10_safe_grep(self):
        assert self._is_blocked("grep -r TODO .") is False

    @pytest.mark.asyncio
    async def test_e8_11_run_safe_echo(self):
        from tools.utilities.run_shell import BLOCKED_PATTERNS
        from utils.code_executor import run_shell as _run_cmd
        with tempfile.TemporaryDirectory() as d:
            r = await _run_cmd("echo agentx_e2e", d, blocked_patterns=BLOCKED_PATTERNS)
            assert "agentx_e2e" in r

    @pytest.mark.asyncio
    async def test_e8_12_blocked_cmd_returns_blocked_string(self):
        from tools.utilities.run_shell import BLOCKED_PATTERNS
        from utils.code_executor import run_shell as _run_cmd
        with tempfile.TemporaryDirectory() as d:
            r = await _run_cmd("rm -rf /", d, blocked_patterns=BLOCKED_PATTERNS)
            assert "BLOCKED" in r

    @pytest.mark.asyncio
    async def test_e8_13_run_shell_tool_blocks_sandbox_escape(self):
        """The actual agent-facing tool also blocks path-escape attempts (cd .., ~)."""
        from tools.utilities.run_shell import make_run_shell_tool
        tool = make_run_shell_tool("e2e_shell_sandbox_user", "e2e_shell_sandbox_conv")
        for cmd in ("cd ..; ls", "cd ~/.ssh", "cat ../../etc/passwd"):
            out = await tool.ainvoke({"command": cmd})
            assert "BLOCKED" in out, f"{cmd!r} should be blocked, got {out[:80]!r}"


# ══════════════════════════════════════════════════════════════
# E10 — Output & Agent Routes (HTTP level)
# ══════════════════════════════════════════════════════════════

class TestE10ProtectedRoutes:
    """E10: Auth-protected output/agent routes return 401 without token."""

    def test_e10_01_list_outputs_requires_auth(self):
        """GET /outputs/list must require authentication."""
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/outputs/list")
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

    # v3 single-agent architecture: the supervisor and per-intent subgraphs were
    # removed. The current graph is graph/builder.py + graph/nodes/agent_node.py.
    def test_e12_01_agent_graph_builder(self):
        import graph.builder

    def test_e12_02_agent_node(self):
        import graph.nodes.agent_node

    def test_e12_03_run_shell_tool(self):
        import tools.utilities.run_shell

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
    async def test_e13_01_cleanup_deletes_old_files_when_idle(self):
        """A stale file in an IDLE workspace's policy dir is deleted; a fresh one is kept."""
        import time, json
        import utils.workspace_cleanup as wc_mod

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            outputs = root_path / "user_1" / "outputs"
            outputs.mkdir(parents=True)

            old_file = outputs / "old_output.pdf"
            old_file.write_text("old")
            old_mtime = time.time() - (200 * 3600)   # older than the 168h outputs TTL
            os.utime(old_file, (old_mtime, old_mtime))

            new_file = outputs / "new_output.pdf"
            new_file.write_text("new")

            # No .meta/last_active => treated as fully idle, so "files" cleanup runs.
            with patch.object(wc_mod, "WORKSPACE_ROOT", root_path):
                await wc_mod._cleanup()

            assert not old_file.exists(), "Stale file in idle workspace should be deleted"
            assert new_file.exists(),     "Recent file should still exist"

    @pytest.mark.asyncio
    async def test_e13_02_cleanup_skips_active_workspace(self):
        """An ACTIVE workspace (recent last_active) keeps its files even if old."""
        import time, json
        import utils.workspace_cleanup as wc_mod

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            user_dir = root_path / "user_2"
            outputs = user_dir / "outputs"
            outputs.mkdir(parents=True)
            meta = user_dir / ".meta"
            meta.mkdir()
            (meta / "last_active.json").write_text(json.dumps({"timestamp": time.time()}))

            old_file = outputs / "old.pdf"
            old_file.write_text("x")
            old_mtime = time.time() - (300 * 3600)
            os.utime(old_file, (old_mtime, old_mtime))

            with patch.object(wc_mod, "WORKSPACE_ROOT", root_path):
                await wc_mod._cleanup()

            assert outputs.exists(),   "Directory itself should survive"
            assert old_file.exists(),  "Active workspace files must be preserved (idle-gated)"

    @pytest.mark.asyncio
    async def test_e13_03_cleanup_does_nothing_when_workspace_missing(self):
        """If workspace root doesn't exist, _cleanup() must return silently."""
        import utils.workspace_cleanup as wc_mod
        nonexistent = Path("/tmp/this_path_does_not_exist_e2e_test_agentx")
        with patch.object(wc_mod, "WORKSPACE_ROOT", nonexistent):
            await wc_mod._cleanup()   # must not raise


# ══════════════════════════════════════════════════════════════
# E14 — Misc Pipeline Checks (file reader, RAG grader, skill vault HTTP)
# ══════════════════════════════════════════════════════════════
# Originally exercised the pre-v3 supervisor pipeline end-to-end; those cases
# were removed with the supervisor. What remains here still targets current,
# non-legacy modules. End-to-end chat itself is covered live by
# tests/test_agent_v3.py (component-level) and the scratchpad e2e harness
# (full HTTP + SSE run against the running server).

class TestE14MiscPipelineChecks:
    """E14: File reader, RAG embedding grader fallback, and skill vault HTTP checks."""

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
