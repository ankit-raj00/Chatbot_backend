"""
Tests for the v3 single-agent architecture and the hardening added around it:
sandbox shell guards, agent_node system-prompt/trim, stateless graph, circuit
breaker, tool cache scoping, and model config. Replaces the removed
supervisor/subgraph tests.
"""
import pytest
import tempfile
from langchain_core.messages import SystemMessage, HumanMessage


# ── Sandbox: run_shell guards ────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", ["rm -rf /", "cd ..; ls", "cat ../../etc/passwd", "cd ~/.ssh", "ls .."])
async def test_run_shell_blocks_dangerous_and_escapes(cmd):
    from tools.utilities.run_shell import make_run_shell_tool
    tool = make_run_shell_tool("test_user_v3", "test_conv_v3")
    out = await tool.ainvoke({"command": cmd})
    assert "BLOCKED" in out, f"{cmd!r} should be blocked, got {out[:80]!r}"


@pytest.mark.asyncio
async def test_run_shell_allows_safe_command():
    from tools.utilities.run_shell import make_run_shell_tool
    tool = make_run_shell_tool("test_user_v3", "test_conv_v3")
    out = await tool.ainvoke({"command": "echo agentx_v3_ok"})
    assert "BLOCKED" not in out and "agentx_v3_ok" in out


# ── agent_node: system prompt + trimming ─────────────────────────────────────

def test_agent_node_system_prompt_merges_and_is_idempotent():
    from graph.nodes.agent_node import _with_system_prompt, AGENT_SYSTEM_PROMPT
    msgs = [SystemMessage(content="BASE PROMPT"), HumanMessage(content="hi")]
    merged = _with_system_prompt(list(msgs))
    assert AGENT_SYSTEM_PROMPT in merged[0].content and "BASE PROMPT" in merged[0].content
    # Running again must not append a second copy.
    merged2 = _with_system_prompt(merged)
    assert merged2[0].content.count(AGENT_SYSTEM_PROMPT) == 1


def test_agent_node_system_prompt_multimodal_content_safe():
    """content as a list of parts (multimodal) must not raise on str concat."""
    from graph.nodes.agent_node import _with_system_prompt, AGENT_SYSTEM_PROMPT
    msgs = [SystemMessage(content=[{"type": "text", "text": "BASE"}]), HumanMessage(content="hi")]
    merged = _with_system_prompt(list(msgs))
    assert AGENT_SYSTEM_PROMPT in merged[0].content


def test_agent_node_trim_bounds_history():
    from graph.nodes.agent_node import _trim, MAX_TURN_MESSAGES
    big = [SystemMessage(content="s")] + [HumanMessage(content=str(i)) for i in range(200)]
    trimmed = _trim(big)
    assert len(trimmed) <= MAX_TURN_MESSAGES + 1  # +1 for the system message


# ── Graph: stateless (no persistent checkpointer) ────────────────────────────

@pytest.mark.asyncio
async def test_agent_graph_has_no_checkpointer():
    from graph.builder import get_agent_graph
    g = await get_agent_graph()
    assert g.checkpointer is None


# ── Circuit breaker: single HALF_OPEN probe ──────────────────────────────────

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_admits_single_probe():
    import asyncio
    from utils.circuit_breaker import CircuitBreaker, ServiceUnavailableError, CircuitState

    cb = CircuitBreaker("t_v3", failure_threshold=1, recovery_timeout=0)

    async def boom():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        await cb.call(boom)
    assert cb.state == CircuitState.OPEN

    async def slow():
        await asyncio.sleep(0.2)
        return "ok"
    probe = asyncio.create_task(cb.call(slow))
    await asyncio.sleep(0.05)
    with pytest.raises(ServiceUnavailableError):
        await cb.call(slow)  # second concurrent probe rejected
    assert await probe == "ok"


# ── Tool cache: user scoping + clock excluded ────────────────────────────────

def test_tool_cache_is_user_scoped_and_excludes_clock():
    from utils.tool_result_cache import _cache_key, CACHEABLE
    assert "get_current_time" not in CACHEABLE
    ka = _cache_key("search_knowledge_base", {"user_id": "A", "q": "x"})
    kb = _cache_key("search_knowledge_base", {"user_id": "B", "q": "x"})
    assert ka != kb and ":A:" in ka and ":B:" in kb


# ── Registry / config ────────────────────────────────────────────────────────

def test_execute_code_not_registered():
    from tools import AVAILABLE_TOOLS
    assert "execute_code" not in AVAILABLE_TOOLS


def test_model_config_pricing_shape():
    from config.model_config import ModelConfig
    p = ModelConfig.get_pricing(ModelConfig.DEFAULT_MODEL)
    assert set(p) == {"input", "output"}
    # real per-token prices now configured (not the placeholder zeros) — not
    # pinning exact values here since prices can change, just that cost
    # computation is no longer a guaranteed-zero no-op.
    assert p["input"] > 0.0
    assert p["output"] > 0.0
    # unknown model -> zeros, never a crash
    z = ModelConfig.get_pricing("does/not-exist")
    assert z == {"input": 0.0, "output": 0.0}
