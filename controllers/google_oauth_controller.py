from fastapi import HTTPException, status
from core.database import oauth_tokens_collection
from datetime import datetime, timedelta
import os
import json
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from typing import Dict, Any, Optional

class GoogleOAuthController:
    """Controller for Google OAuth authentication flow"""
    
    # Configuration
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/userinfo.email',
        'https://www.googleapis.com/auth/userinfo.profile',
        'openid'
    ]
    # In production, these should be env vars, but for now we'll load from client_secrets.json
    CLIENT_SECRETS_FILE = "client_secrets.json"
    
    @staticmethod
    def _get_flow(redirect_uri: str):
        """Create OAuth flow instance from env vars or file"""
        try:
            # Check for env vars first
            client_id = os.getenv("CLIENT_ID")
            client_secret = os.getenv("CLIENT_SECRET")
            
            if client_id and client_secret:
                client_config = {
                    "web": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [redirect_uri]
                    }
                }
                return Flow.from_client_config(
                    client_config,
                    scopes=GoogleOAuthController.SCOPES,
                    redirect_uri=redirect_uri
                )
            
            # Fallback to file
            return Flow.from_client_secrets_file(
                GoogleOAuthController.CLIENT_SECRETS_FILE,
                scopes=GoogleOAuthController.SCOPES,
                redirect_uri=redirect_uri
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load client secrets: {str(e)}"
            )

    @staticmethod
    async def initiate_oauth(user_id: str, redirect_uri: str):
        """Initiate Google OAuth flow"""
        try:
            flow = GoogleOAuthController._get_flow(redirect_uri)
            
            # Generate authorization URL
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent',
                state=user_id  # Pass user_id as state to recover it in callback
            )
            
            return {
                "oauth_url": auth_url,
                "state": state
            }
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @staticmethod
    async def handle_callback(code: str, state: str, redirect_uri: str):
        """Handle OAuth callback and exchange code for tokens"""
        try:
            user_id = state  # We passed user_id as state
            
            flow = GoogleOAuthController._get_flow(redirect_uri)
            flow.fetch_token(code=code)
            creds = flow.credentials
            
            # Get user email from Google
            from googleapiclient.discovery import build
            service = build('oauth2', 'v2', credentials=creds)
            user_info = service.userinfo().get().execute()
            email = user_info.get('email', 'Unknown')
            
            # Store tokens in database
            token_data = {
                "user_id": user_id,
                "provider": "google",
                "email": email,
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
                "expiry": creds.expiry.isoformat() if creds.expiry else None,
                "updated_at": datetime.now()
            }
            
            # Update or insert
            await oauth_tokens_collection.update_one(
                {"user_id": user_id, "provider": "google"},
                {"$set": token_data},
                upsert=True
            )
            
            return {
                "success": True,
                "message": "Google authentication successful"
            }
            
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @staticmethod
    async def get_user_credentials(user_id: str) -> Optional[Credentials]:
        """Get valid user credentials, refreshing if necessary"""
        try:
            token_doc = await oauth_tokens_collection.find_one({
                "user_id": user_id,
                "provider": "google"
            })
            
            if not token_doc:
                return None
                
            # Reconstruct credentials
            creds = Credentials(
                token=token_doc.get("token"),
                refresh_token=token_doc.get("refresh_token"),
                token_uri=token_doc.get("token_uri"),
                client_id=token_doc.get("client_id"),
                client_secret=token_doc.get("client_secret"),
                scopes=token_doc.get("scopes"),
                expiry=datetime.fromisoformat(token_doc.get("expiry")) if token_doc.get("expiry") else None
            )
            
            # Refresh if expired
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                
                # Update in DB
                await oauth_tokens_collection.update_one(
                    {"user_id": user_id, "provider": "google"},
                    {"$set": {
                        "token": creds.token,
                        "expiry": creds.expiry.isoformat() if creds.expiry else None,
                        "updated_at": datetime.now()
                    }}
                )
            
            return creds
            
        except Exception as e:
            print(f"Error getting user credentials: {e}")
            return None
