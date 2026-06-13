"""
UserSkill — schema for user-uploaded skill vault entries.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UserSkillCreate(BaseModel):
    skill_name: str
    skill_content: str          # Raw SKILL.md content (frontmatter + body)
    description: Optional[str] = None


class UserSkillOut(BaseModel):
    id: str
    user_id: str
    skill_name: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
