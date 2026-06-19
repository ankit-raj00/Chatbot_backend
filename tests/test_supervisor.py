"""Tests for the supervisor graph — intent classification and routing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage

# Correct patch targets: get_llm is imported inside the node function from
# graph.llm_registry, so we patch it at the source module.
LLM_PATCH = "graph.llm_registry.get_llm"
SKILL_PATCH = "skills.skill_loader.get_relevant_skill_for_message"


def _base_state(**overrides):
    s = {
        "messages": [HumanMessage(content="Hello")],
        "user_id": "u1", "conversation_id": "c1",
        "agent": "", "model": "gemini-2.5-flash",
        "enabled_tools": [], "selected_files": None,
        "skill_body": "", "final_response": "",
    }
    s.update(overrides)
    return s


@pytest.mark.asyncio
async def test_intent_classifier_chat():
    """General questions should route to chat."""
    from graph.supervisor import intent_classifier_node

    state = _base_state(messages=[HumanMessage(content="What is machine learning?")])
    mock_response = MagicMock()
    mock_response.content = "chat"

    with patch(LLM_PATCH) as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        with patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            result = await intent_classifier_node(state)

    assert result["agent"] == "chat"
    assert "skill_body" in result


@pytest.mark.asyncio
async def test_intent_classifier_document():
    """PDF requests should route to document."""
    from graph.supervisor import intent_classifier_node

    state = _base_state(messages=[HumanMessage(content="Create a PDF report on sales data")])
    mock_response = MagicMock()
    mock_response.content = "document"

    with patch(LLM_PATCH) as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        with patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            result = await intent_classifier_node(state)

    assert result["agent"] == "document"


@pytest.mark.asyncio
async def test_intent_classifier_invalid_falls_back_to_chat():
    """Invalid agent name from LLM should fall back to chat."""
    from graph.supervisor import intent_classifier_node

    state = _base_state(messages=[HumanMessage(content="Hello")])
    mock_response = MagicMock()
    mock_response.content = "invalid_agent_xyz"

    with patch(LLM_PATCH) as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        with patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            result = await intent_classifier_node(state)

    assert result["agent"] == "chat"


@pytest.mark.asyncio
async def test_intent_classifier_api_failure_falls_back():
    """If LLM call fails, should fall back to chat."""
    from graph.supervisor import intent_classifier_node

    state = _base_state(messages=[HumanMessage(content="Hello")])

    with patch(LLM_PATCH) as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
        mock_get_llm.return_value = mock_llm

        with patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            result = await intent_classifier_node(state)

    assert result["agent"] == "chat"


@pytest.mark.asyncio
async def test_intent_classifier_empty_messages():
    """Empty messages list should default to chat immediately."""
    from graph.supervisor import intent_classifier_node

    state = _base_state(messages=[])
    result = await intent_classifier_node(state)
    assert result["agent"] == "chat"


def test_valid_agents_set():
    """The VALID_AGENTS set must have exactly the 7 expected agents."""
    from graph.supervisor import VALID_AGENTS
    expected = {"chat", "shell", "document", "vision", "code", "rag", "data"}
    assert VALID_AGENTS == expected


@pytest.mark.asyncio
async def test_intent_classifier_shell():
    """Run script requests should route to shell."""
    from graph.supervisor import intent_classifier_node

    state = _base_state(messages=[HumanMessage(content="run my python script")])
    mock_response = MagicMock()
    mock_response.content = "shell"

    with patch(LLM_PATCH) as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        with patch(SKILL_PATCH, new_callable=AsyncMock, return_value=None):
            result = await intent_classifier_node(state)

    assert result["agent"] == "shell"
