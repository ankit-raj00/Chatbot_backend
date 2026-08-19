import os

from fastapi import APIRouter, Depends, Request

from controllers.auth_controller import AuthController
from models.user import UserCreate, UserLogin
from core.middleware import get_current_user
from core.limiter import limiter

router = APIRouter(prefix="/auth", tags=["Authentication"])

auth_controller = AuthController()


# Rate limits here are load-bearing, not decoration — this app is publicly
# reachable and these two endpoints are the ones worth abusing.
#
# signup: every new account is granted DEFAULT_CREDIT_CAP_USD (5.0) of free
# LLM credit, so an unthrottled signup endpoint is a direct route to burning
# real money — register in a loop, spend the grant, repeat. The cap is
# per-user, so it only bounds spend if creating users is itself bounded.
#
# login: unthrottled failed logins are free credential-stuffing attempts
# against every account on the system.
#
# Limits are per client IP (see core/limiter.py). They are deliberately
# generous enough not to interfere with a real person mistyping a password
# or signing up from a shared/NAT'd network.
SIGNUP_RATE_LIMIT = os.getenv("SIGNUP_RATE_LIMIT", "5/hour")
LOGIN_RATE_LIMIT = os.getenv("LOGIN_RATE_LIMIT", "10/minute")


@router.post("/signup")
@limiter.limit(SIGNUP_RATE_LIMIT)
async def signup(request: Request, user_data: UserCreate):
    """Register a new user"""
    return await auth_controller.signup(user_data)


@router.post("/login")
@limiter.limit(LOGIN_RATE_LIMIT)
async def login(request: Request, credentials: UserLogin):
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
