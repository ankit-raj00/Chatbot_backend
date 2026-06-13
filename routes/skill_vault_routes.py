"""
Skill Vault Routes — user uploads, lists, validates and uses their own SKILL.md files.

Endpoints:
  GET    /api/skills/vault              — list user's uploaded skills
  POST   /api/skills/vault              — upload a new skill (SKILL.md content)
  DELETE /api/skills/vault/{skill_name} — delete a skill
  POST   /api/skills/vault/validate     — validate SKILL.md frontmatter without saving
  GET    /api/skills/vault/{skill_name} — preview a skill's full content
  GET    /api/skills/builtin            — list all built-in system skills
"""

import re
from datetime import datetime
from typing import Optional
from bson import ObjectId

from fastapi import APIRouter, Depends, HTTPException
import yaml

from core.database import user_skills_collection
from core.middleware import get_current_user
from models.user_skill import UserSkillCreate

router = APIRouter(prefix="/api/skills", tags=["Skills"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter. Returns (meta_dict, body)."""
    m = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not m:
        return {}, content
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        body = content[m.end():]
        return meta, body
    except yaml.YAMLError as e:
        raise ValueError(str(e))


def _validate_skill(content: str) -> list[str]:
    """Validate a SKILL.md file. Returns a list of error strings (empty = valid)."""
    errors = []
    try:
        meta, body = _parse_frontmatter(content)
    except ValueError as e:
        return [f"YAML parse error: {e}"]

    if not meta.get("name"):
        errors.append("Missing required field: name")
    if not meta.get("description"):
        errors.append("Missing required field: description")
    if len(body.strip()) < 50:
        errors.append("Skill body must have at least 50 characters")
    return errors


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/builtin")
async def list_builtin_skills():
    """List all built-in system skills."""
    from skills.skill_loader import list_builtin_skills as _list
    return {"skills": _list()}


@router.get("/vault")
async def list_vault_skills(current_user: dict = Depends(get_current_user)):
    """List all skills uploaded to the user's vault."""
    from skills.skill_loader import list_user_skills
    user_id = str(current_user["_id"])
    skills = await list_user_skills(user_id)
    return {"skills": skills}


@router.post("/vault")
async def upload_skill(
    payload: UserSkillCreate,
    current_user: dict = Depends(get_current_user),
):
    """Upload a new SKILL.md to the user's vault."""
    user_id = str(current_user["_id"])

    errors = _validate_skill(payload.skill_content)
    if errors:
        raise HTTPException(status_code=422, detail={"valid": False, "errors": errors})

    meta, _ = _parse_frontmatter(payload.skill_content)
    skill_name = payload.skill_name or meta.get("name", "unnamed")
    description = payload.description or meta.get("description", "")
    triggers = (meta.get("metadata") or {}).get("triggers", [])
    agent = (meta.get("metadata") or {}).get("agent", "chat")

    # Upsert: if same skill_name exists, replace it
    result = await user_skills_collection.update_one(
        {"user_id": user_id, "skill_name": skill_name},
        {"$set": {
            "user_id":      user_id,
            "skill_name":   skill_name,
            "description":  description,
            "skill_content": payload.skill_content,
            "triggers":     triggers,
            "agent":        agent,
            "is_active":    True,
            "created_at":   datetime.utcnow(),
        }},
        upsert=True,
    )
    return {
        "success": True,
        "skill_name": skill_name,
        "upserted": result.upserted_id is not None,
    }


@router.post("/vault/validate")
async def validate_skill(payload: UserSkillCreate):
    """Validate a SKILL.md without saving it."""
    errors = _validate_skill(payload.skill_content)
    if errors:
        return {"valid": False, "errors": errors}
    meta, _ = _parse_frontmatter(payload.skill_content)
    return {"valid": True, "name": meta.get("name"), "description": meta.get("description")}


@router.get("/vault/{skill_name}")
async def get_skill(skill_name: str, current_user: dict = Depends(get_current_user)):
    """Preview the full content of a user's skill."""
    user_id = str(current_user["_id"])
    doc = await user_skills_collection.find_one(
        {"user_id": user_id, "skill_name": skill_name}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Skill not found")
    doc["_id"] = str(doc["_id"])
    return doc


@router.delete("/vault/{skill_name}")
async def delete_skill(skill_name: str, current_user: dict = Depends(get_current_user)):
    """Delete a skill from the user's vault."""
    user_id = str(current_user["_id"])
    result = await user_skills_collection.delete_one(
        {"user_id": user_id, "skill_name": skill_name}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"success": True, "skill_name": skill_name}
