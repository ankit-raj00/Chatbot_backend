"""Tests for specialist subgraphs — chat, shell sandbox safety, data subgraph."""
import sys, os, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# mcp_manager is imported inside the subgraph body, so patch at the source module
LLM_PATCH = "graph.llm_registry.get_llm"
MCP_PATCH = "utils.mcp_connection_manager.mcp_manager"


def _base_state(**overrides):
    state = {
        "messages": [HumanMessage(content="Hello")],
        "user_id": "test_user_123",
        "conversation_id": "conv_abc",
        "agent": "chat",
        "model": "gemini-2.5-flash",
        "enabled_tools": [],
        "selected_files": None,
        "skill_body": "",
        "final_response": "",
    }
    state.update(overrides)
    return state


# ── Chat Subgraph ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_subgraph_returns_response():
    from graph.subgraphs.chat_subgraph import chat_subgraph

    mock_ai = AIMessage(content="Hello there!")
    with patch(LLM_PATCH) as mock_get_llm, \
         patch(MCP_PATCH) as mock_mcp:

        mock_mcp.get_all_langchain_tools = AsyncMock(return_value=[])
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        result = await chat_subgraph(_base_state(), {})

    assert result["final_response"] == "Hello there!"
    assert len(result["messages"]) == 1


@pytest.mark.asyncio
async def test_chat_subgraph_injects_skill_into_system_message():
    """When skill_body is set, it should be prepended as a SystemMessage."""
    from graph.subgraphs.chat_subgraph import chat_subgraph

    mock_ai = AIMessage(content="I will help with that PDF.")
    with patch(LLM_PATCH) as mock_get_llm, \
         patch(MCP_PATCH) as mock_mcp:

        mock_mcp.get_all_langchain_tools = AsyncMock(return_value=[])
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        state = _base_state(skill_body="# PDF Guide\nUse reportlab for PDFs.")
        await chat_subgraph(state, {})

    # Verify that ainvoke received messages with a SystemMessage as first item
    mock_llm.ainvoke.assert_called_once()
    called_messages = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(called_messages[0], SystemMessage), \
        f"Expected SystemMessage, got {type(called_messages[0])}"
    assert "PDF Guide" in called_messages[0].content


@pytest.mark.asyncio
async def test_chat_subgraph_existing_system_message_gets_skill_appended():
    """When messages already start with a SystemMessage, skill_body should be appended."""
    from graph.subgraphs.chat_subgraph import chat_subgraph

    mock_ai = AIMessage(content="Sure!")
    with patch(LLM_PATCH) as mock_get_llm, \
         patch(MCP_PATCH) as mock_mcp:

        mock_mcp.get_all_langchain_tools = AsyncMock(return_value=[])
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        state = _base_state(
            messages=[SystemMessage(content="You are helpful."), HumanMessage(content="Help!")],
            skill_body="## SKILL\nDo this.",
        )
        await chat_subgraph(state, {})

    called_messages = mock_llm.ainvoke.call_args[0][0]
    # First message should still be a SystemMessage with both original + skill
    assert isinstance(called_messages[0], SystemMessage)
    assert "You are helpful." in called_messages[0].content
    assert "SKILL" in called_messages[0].content


# ── Shell Subgraph Safety ──────────────────────────────────────────────────────

def test_is_blocked_dangerous():
    """Dangerous commands must be blocked (exact patterns from BLOCKED list)."""
    from graph.subgraphs.shell_subgraph import _is_blocked
    assert _is_blocked("rm -rf /") is True
    assert _is_blocked("sudo rm -rf ~") is True
    # The BLOCKED list uses 'curl | sh' and 'wget | sh' (no URL between)
    assert _is_blocked("curl | sh") is True
    assert _is_blocked("wget | bash") is True
    assert _is_blocked(":(){:|:&};:") is True
    assert _is_blocked("mkfs /dev/sda") is True


def test_is_blocked_safe():
    """Safe commands must NOT be blocked."""
    from graph.subgraphs.shell_subgraph import _is_blocked
    assert _is_blocked("ls -la") is False
    assert _is_blocked("cat README.md") is False
    assert _is_blocked("python test.py") is False
    assert _is_blocked("grep -r 'TODO' .") is False


@pytest.mark.asyncio
async def test_run_cmd_safe_echo():
    """A safe echo command must succeed."""
    from graph.subgraphs.shell_subgraph import _run_cmd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await _run_cmd("echo hello_world", tmpdir)
        assert "hello_world" in result


@pytest.mark.asyncio
async def test_run_cmd_blocked_returns_blocked():
    """A blocked command must return the BLOCKED string, not raise."""
    from graph.subgraphs.shell_subgraph import _run_cmd
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await _run_cmd("rm -rf /", tmpdir)
        assert "BLOCKED" in result


# ── Data Subgraph ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_file_reader_csv_used_by_data_subgraph():
    """The universal file reader (used by the data agent) must handle CSV."""
    from pathlib import Path
    from services.universal_file_reader import extract_any_file

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                      delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "score"])
        writer.writeheader()
        writer.writerows([{"name": "Alice", "score": "95"},
                          {"name": "Bob",   "score": "82"}])
        fname = f.name

    try:
        result = await extract_any_file(Path(fname))
        assert result["type"] == "csv", f"Expected csv, got: {result}"
        assert result["row_count"] == 2
        assert "name" in result["columns"]
    finally:
        os.unlink(fname)
