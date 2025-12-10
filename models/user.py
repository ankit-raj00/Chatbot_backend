from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class User(BaseModel):
    """User model for database"""
    id: Optional[str] = Field(None, alias="_id")
    email: EmailStr
    name: str
    password: str  # Hashed password
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class UserCreate(BaseModel):
    """Request model for user signup"""
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    """Request model for user login"""
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """Response model for user data (without password)"""
    id: str = Field(alias="_id")
    email: EmailStr
    name: str
    created_at: datetime
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Data stored in JWT token"""
    user_id: str
    email: str
