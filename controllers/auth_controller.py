from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from core.database import users_collection
from models.user import UserCreate, UserLogin
from core.auth import verify_password, get_password_hash, create_access_token
from datetime import datetime
from bson import ObjectId

class AuthController:
    """Controller for authentication operations"""
    
    @staticmethod
    async def signup(user_data: UserCreate):
        """Register a new user and set JWT in HTTP-only cookie"""
        try:
            # Check if user already exists
            existing_user = await users_collection.find_one({"email": user_data.email})
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password
            hashed_password = get_password_hash(user_data.password)
            
            # Create user document
            new_user = {
                "email": user_data.email,
                "name": user_data.name,
                "password": hashed_password,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Insert user
            result = await users_collection.insert_one(new_user)
            user_id = str(result.inserted_id)
            
            # Create JWT token
            access_token = create_access_token(
                data={"user_id": user_id, "email": user_data.email}
            )
            
            # Create response with user data
            response = JSONResponse(content={
                "user": {
                    "id": user_id,
                    "name": user_data.name,
                    "email": user_data.email
                },
                "message": "User created successfully"
            })
            
            # Set HTTP-only cookie
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite="lax",
                max_age=30 * 24 * 60 * 60  # 30 days
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Signup error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def login(credentials: UserLogin):
        """Login user and set JWT in HTTP-only cookie"""
        try:
            # Find user by email
            user = await users_collection.find_one({"email": credentials.email})
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Verify password
            if not verify_password(credentials.password, user["password"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Create JWT token
            access_token = create_access_token(
                data={"user_id": str(user["_id"]), "email": user["email"]}
            )
            
            # Create response with user data
            response = JSONResponse(content={
                "user": {
                    "id": str(user["_id"]),
                    "name": user["name"],
                    "email": user["email"]
                },
                "message": "Login successful"
            })
            
            # Set HTTP-only cookie
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite="lax",
                max_age=30 * 24 * 60 * 60  # 30 days
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Login error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @staticmethod
    async def logout():
        """Logout user by clearing the cookie"""
        response = JSONResponse(content={"message": "Logged out successfully"})
        response.delete_cookie(key="access_token")
        return response
    
    @staticmethod
    async def get_current_user_info(current_user: dict):
        """Get current user information"""
        return {
            "id": str(current_user["_id"]),
            "email": current_user["email"],
            "name": current_user["name"],
            "created_at": current_user["created_at"]
        }
