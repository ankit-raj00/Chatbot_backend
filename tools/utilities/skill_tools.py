"""
Skill tools — expose the skill vault as agent-callable tools.

list_skills:  Level 1 — name + description only (the "menu")
load_skill:   Level 2 — full SKILL.md body, loaded on demand
"""
from typing import Optional
from langchain_core.tools import tool
from skills.skill_loader import (
    list_builtin_skills,
    load_builtin_skill,
    load_user_skill,
    list_user_skills,
)


@tool
def list_skills() -> str:
    """
    List all available skills (built-in + your uploaded vault skills) with
    a short description of what each one covers.

    Call this if you're unsure which skills exist or which one applies —
    it costs almost nothing and helps you pick the right manual before
    starting a complex task (e.g. creating documents, analyzing data,
    generating diagrams, reviewing code).
    """
    builtin = list_builtin_skills()
    lines = ["## Built-in skills"]
    for s in builtin:
        lines.append(f"- **{s['name']}** ({s.get('agent','')}): {s['description'][:150]}")
    return "\n".join(lines)


def make_load_skill_tool(user_id: str):
    @tool
    async def load_skill(skill_name: str) -> str:
        """
        Load the full instructions ("manual") for a named skill.

        Call this BEFORE attempting a task that a skill covers (e.g. before
        writing a PDF-generation script, call load_skill("create-pdf") to
        see the recommended libraries, output conventions, and gotchas).

        Checks built-in skills first, then your uploaded vault skills.

        Args:
            skill_name: The skill's name, e.g. "create-pdf", "create-docx",
                        "analyze-data", "code-review", "generate-diagram".
        """
        body = load_builtin_skill(skill_name)
        if body:
            return body
        user_body = await load_user_skill(user_id, skill_name)
        if user_body:
            return user_body
        return (
            f"No skill named '{skill_name}' found. "
            f"Call list_skills() to see available skills."
        )

    return load_skill
