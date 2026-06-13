"""
Skill Loader — the entire "skills system" in one file.

Reads SKILL.md content and returns it for injection into agent system prompts.
For user-uploaded skills: reads from MongoDB instead of filesystem.

Three-level loading (mirrors the real skill system architecture):
  Level 1 — Skill listing in system prompt (name + description only)
  Level 2 — Full SKILL.md body loaded on demand by subgraph nodes
  Level 3 — References/scripts loaded on demand when needed
"""

import re
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

BUILTIN_DIR = Path(__file__).parent / "builtin"


def load_builtin_skill(skill_name: str) -> Optional[str]:
    """
    Load a built-in skill's SKILL.md content (body only, strips frontmatter).
    Returns None if skill not found.

    Usage in an agent node system prompt:
        skill = load_builtin_skill("create-pdf")
        system_prompt = f"You are a document agent.\\n\\n{skill}\\n\\nNow help the user."
    """
    skill_name = skill_name.replace("_", "-")   # Normalize underscores to kebab-case
    skill_path = BUILTIN_DIR / skill_name / "SKILL.md"
    if not skill_path.exists():
        logger.warning(f"skill_loader.not_found skill={skill_name}")
        return None
    content = skill_path.read_text(encoding="utf-8")
    body = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL).strip()
    logger.info(f"skill_loader.loaded skill={skill_name} chars={len(body)}")
    return body


def list_builtin_skills() -> list[dict]:
    """List all available built-in skills with their metadata from frontmatter."""
    import yaml
    skills = []
    if not BUILTIN_DIR.exists():
        return skills
    for skill_dir in sorted(BUILTIN_DIR.iterdir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        content = skill_file.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if m:
            try:
                meta = yaml.safe_load(m.group(1))
                metadata_block = meta.get("metadata", {}) or {}
                skills.append({
                    "name":        meta.get("name", skill_dir.name),
                    "description": meta.get("description", ""),
                    "agent":       metadata_block.get("agent", ""),
                    "triggers":    metadata_block.get("triggers", []),
                })
            except Exception:
                pass
    return skills


async def load_user_skill(user_id: str, skill_name: str) -> Optional[str]:
    """
    Load a user-uploaded skill from MongoDB.
    Returns the skill body (frontmatter stripped), or None if not found.
    """
    from core.database import user_skills_collection
    doc = await user_skills_collection.find_one({
        "user_id": user_id,
        "skill_name": skill_name,
        "is_active": True,
    })
    if not doc:
        return None
    content = doc.get("skill_content", "")
    body = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL).strip()
    return body


async def list_user_skills(user_id: str) -> list[dict]:
    """List all active user-uploaded skills."""
    from core.database import user_skills_collection
    cursor = user_skills_collection.find(
        {"user_id": user_id, "is_active": True},
        {"skill_content": 0}   # exclude raw content from listing
    )
    docs = await cursor.to_list(100)
    for d in docs:
        d["_id"] = str(d["_id"])
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
    return docs


async def get_relevant_skill_for_message(
    message: str,
    user_id: str = "",
    agent_type: str = "",
) -> Optional[tuple[str, str]]:
    """
    Given a user message, find and return the most relevant skill body.
    Checks:
      1. Builtin skills whose metadata.triggers match words in the message
      2. User vault skills whose triggers match

    Returns (body, name) or None if no match.
    """
    import yaml

    message_lower = message.lower()

    # 1. Check builtin skills
    if BUILTIN_DIR.exists():
        for skill_dir in sorted(BUILTIN_DIR.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            content = skill_file.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
            if not m:
                continue
            try:
                meta = yaml.safe_load(m.group(1))
                skill_agent = (meta.get("metadata") or {}).get("agent", "")
                if agent_type and skill_agent and skill_agent != agent_type:
                    continue
                triggers = (meta.get("metadata") or {}).get("triggers", [])
                
                # Check if all words in ANY trigger are present in the message
                message_words = set(message_lower.split())
                is_match = False
                for t in triggers:
                    trigger_words = set(t.lower().split())
                    if trigger_words.issubset(message_words):
                        is_match = True
                        break

                if is_match:
                    body = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL).strip()
                    logger.info(f"skill_loader.triggered skill={skill_dir.name} agent={agent_type}")
                    return body, skill_dir.name
            except Exception:
                continue

    # 2. Check user vault skills
    if user_id:
        try:
            from core.database import user_skills_collection
            cursor = user_skills_collection.find(
                {"user_id": user_id, "is_active": True},
                {"skill_content": 1, "triggers": 1, "agent": 1, "skill_name": 1}
            )
            async for doc in cursor:
                skill_agent = doc.get("agent", "")
                if agent_type and skill_agent and skill_agent != agent_type:
                    continue
                triggers = doc.get("triggers", [])
                
                message_words = set(message_lower.split())
                is_match = False
                for t in triggers:
                    trigger_words = set(t.lower().split())
                    if trigger_words.issubset(message_words):
                        is_match = True
                        break

                if is_match:
                    content = doc.get("skill_content", "")
                    body = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL).strip()
                    logger.info(f"skill_loader.user_skill_triggered skill={doc['skill_name']}")
                    return body, doc['skill_name']
        except Exception:
            pass

    return None


def get_skill_scripts_path(skill_name: str) -> Optional[Path]:
    """Returns the path to a skill's scripts/ directory if it exists."""
    skill_name = skill_name.replace("_", "-")
    scripts_dir = BUILTIN_DIR / skill_name / "scripts"
    return scripts_dir if scripts_dir.exists() else None


def get_skill_references_path(skill_name: str) -> Optional[Path]:
    """Returns the path to a skill's references/ directory."""
    skill_name = skill_name.replace("_", "-")
    refs_dir = BUILTIN_DIR / skill_name / "references"
    return refs_dir if refs_dir.exists() else None
