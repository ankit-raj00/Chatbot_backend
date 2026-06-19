"""Tests for the skill loader system."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from skills.skill_loader import (
    list_builtin_skills,
    load_builtin_skill,
    get_relevant_skill_for_message,
)


def test_list_builtin_skills_returns_list():
    skills = list_builtin_skills()
    assert isinstance(skills, list)
    assert len(skills) >= 14, f"Expected >=14 skills, got {len(skills)}"


def test_all_skills_have_required_fields():
    skills = list_builtin_skills()
    for s in skills:
        assert "name" in s and s["name"], f"Skill missing name: {s}"
        assert "description" in s and s["description"], f"Skill missing description: {s}"
        assert "agent" in s, f"Skill missing agent: {s}"
        assert "triggers" in s, f"Skill missing triggers: {s}"


def test_load_builtin_skill_returns_body():
    body = load_builtin_skill("create-pdf")
    assert body is not None
    assert len(body) > 100, "PDF skill body should have substantial content"
    # Should not contain frontmatter markers
    assert "---" not in body[:10], "Body should not start with frontmatter"


def test_load_builtin_skill_not_found():
    body = load_builtin_skill("nonexistent-skill-xyz")
    assert body is None


def test_load_builtin_skill_underscore_normalization():
    """Underscores should be normalized to kebab-case."""
    body = load_builtin_skill("create_pdf")
    assert body is not None


@pytest.mark.asyncio
async def test_get_relevant_skill_for_message_pdf():
    skill_res = await get_relevant_skill_for_message("create a pdf report for Q3")
    assert skill_res is not None, "Should match create-pdf skill"
    body, name = skill_res
    assert len(body) > 50
    assert name == "create-pdf"


@pytest.mark.asyncio
async def test_get_relevant_skill_for_message_agent_filter():
    # Ask for a 'chat' agent skill when using 'document' filter — should not match pdf
    skill_res = await get_relevant_skill_for_message("create a pdf report", agent_type="shell")
    # create-pdf has agent=document, so it should not match when filtering for shell
    # (unless there's a shell skill that also matches)
    assert skill_res is None or "shell" in skill_res[1].lower() or "read" in skill_res[1].lower()


@pytest.mark.asyncio
async def test_get_relevant_skill_for_message_no_match():
    skill = await get_relevant_skill_for_message("what is 2 + 2")
    # Generic math question — no skill should trigger
    assert skill is None
