from fastapi import APIRouter, Depends
from controllers.auth_controller import AuthController
from models.user import UserCreate, UserLogin
from core.middleware import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

auth_controller = AuthController()

@router.post("/signup")
async def signup(user_data: UserCreate):
    """Register a new user"""
    return await auth_controller.signup(user_data)

@router.post("/login")
async def login(credentials: UserLogin):
    """Login user"""
    return await auth_controller.login(credentials)

@router.post("/logout")
async def logout():
    """Logout user"""
    return await auth_controller.logout()

@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return await auth_controller.get_current_user_info(current_user)
