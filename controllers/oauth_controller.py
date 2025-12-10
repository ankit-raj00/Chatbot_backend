from fastapi import HTTPException, status
from core.database import oauth_tokens_collection
from datetime import datetime, timedelta
from bson import ObjectId
import secrets
import httpx
from typing import Dict, Any

class OAuthController:
    """Controller for OAuth authentication flow"""
    
    # In-memory storage for OAuth states (in production, use Redis)
    oauth_states: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    async def initiate_oauth(server_id: str, user_id: str, redirect_uri: str):
        """Initiate OAuth flow for an MCP server"""
        try:
            # Get server
            server = await mcp_servers_collection.find_one({
                "_id": ObjectId(server_id),
                "user_id": user_id
            })
            if not server:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
            
            if server.get("auth_type") != "oauth":
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Server does not use OAuth")
            
            oauth_config = server.get("oauth_config", {})
            if not oauth_config:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth config not found")
            
            # Generate state for CSRF protection
            state = secrets.token_urlsafe(32)
            
            # Store state with server info
            OAuthController.oauth_states[state] = {
                "server_id": server_id,
                "user_id": user_id,
                "created_at": datetime.now(),
                "redirect_uri": redirect_uri
            }
            
            # Build OAuth URL
            auth_url = oauth_config.get("auth_url")
            client_id = oauth_config.get("client_id")
            scopes = oauth_config.get("scopes", "")
            
            oauth_url = (
                f"{auth_url}?"
                f"client_id={client_id}&"
                f"redirect_uri={redirect_uri}&"
                f"response_type=code&"
                f"state={state}&"
                f"scope={scopes}"
            )
            
            return {
                "oauth_url": oauth_url,
                "state": state
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    @staticmethod
    async def handle_callback(code: str, state: str):
        """Handle OAuth callback and exchange code for tokens"""
        try:
            # Validate state
            if state not in OAuthController.oauth_states:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid state")
            
            state_data = OAuthController.oauth_states[state]
            
            # Check if state is expired (10 minutes)
            if datetime.now() - state_data["created_at"] > timedelta(minutes=10):
                del OAuthController.oauth_states[state]
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="State expired")
            
            server_id = state_data["server_id"]
            user_id = state_data["user_id"]
            redirect_uri = state_data["redirect_uri"]
            
            # Get server
            server = await mcp_servers_collection.find_one({
                "_id": ObjectId(server_id),
                "user_id": user_id
            })
            if not server:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
            
            oauth_config = server.get("oauth_config", {})
            token_url = oauth_config.get("token_url")
            client_id = oauth_config.get("client_id")
            client_secret = oauth_config.get("client_secret")
            
            # Exchange code for tokens
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": redirect_uri,
                        "client_id": client_id,
                        "client_secret": client_secret
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Token exchange failed: {response.text}"
                    )
                
                tokens = response.json()
            
            # Store tokens in database
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")
            expires_in = tokens.get("expires_in", 3600)
            
            await mcp_servers_collection.update_one(
                {"_id": ObjectId(server_id)},
                {"$set": {
                    "access_token": access_token,  # TODO: Encrypt in production
                    "refresh_token": refresh_token,  # TODO: Encrypt in production
                    "token_expires_at": datetime.now() + timedelta(seconds=expires_in),
                    "updated_at": datetime.now()
                }}
            )
            
            # Clean up state
            del OAuthController.oauth_states[state]
            
            return {
                "success": True,
                "server_id": server_id,
                "message": "Authentication successful"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    @staticmethod
    async def refresh_token(server_id: str, user_id: str):
        """Refresh expired access token"""
        try:
            server = await mcp_servers_collection.find_one({
                "_id": ObjectId(server_id),
                "user_id": user_id
            })
            if not server:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server not found")
            
            refresh_token = server.get("refresh_token")
            if not refresh_token:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No refresh token available")
            
            oauth_config = server.get("oauth_config", {})
            token_url = oauth_config.get("token_url")
            client_id = oauth_config.get("client_id")
            client_secret = oauth_config.get("client_secret")
            
            # Refresh token
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": client_id,
                        "client_secret": client_secret
                    }
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Token refresh failed: {response.text}"
                    )
                
                tokens = response.json()
            
            # Update tokens
            access_token = tokens.get("access_token")
            new_refresh_token = tokens.get("refresh_token", refresh_token)
            expires_in = tokens.get("expires_in", 3600)
            
            await mcp_servers_collection.update_one(
                {"_id": ObjectId(server_id)},
                {"$set": {
                    "access_token": access_token,
                    "refresh_token": new_refresh_token,
                    "token_expires_at": datetime.now() + timedelta(seconds=expires_in),
                    "updated_at": datetime.now()
                }}
            )
            
            return {"success": True, "message": "Token refreshed"}
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
