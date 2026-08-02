"""
MongoDB-backed TokenStorage for MCP OAuth 2.1 connections.

Implements the `mcp.client.auth.oauth2.TokenStorage` protocol (get/set tokens,
get/set client_info) per (user_id, server_id) pair. Tokens and dynamically-
registered client credentials (which may include a client_secret) are encrypted
at rest via utils/crypto.py — the whole JSON blob is encrypted as one opaque
string rather than field-by-field, since it's a single credential bundle.
"""
from datetime import datetime, timezone

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from core.database import mcp_oauth_tokens_collection
from utils.crypto import encrypt_str, decrypt_str


class MongoTokenStorage:
    """TokenStorage for a single (user_id, server_id) MCP OAuth connection."""

    def __init__(self, user_id: str, server_id: str):
        self.user_id = user_id
        self.server_id = server_id

    def _key(self) -> dict:
        return {"user_id": self.user_id, "server_id": self.server_id}

    async def get_tokens(self) -> OAuthToken | None:
        doc = await mcp_oauth_tokens_collection.find_one(self._key())
        if not doc or not doc.get("tokens_enc"):
            return None
        return OAuthToken.model_validate_json(decrypt_str(doc["tokens_enc"]))

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await mcp_oauth_tokens_collection.update_one(
            self._key(),
            {"$set": {
                "tokens_enc": encrypt_str(tokens.model_dump_json()),
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        doc = await mcp_oauth_tokens_collection.find_one(self._key())
        if not doc or not doc.get("client_info_enc"):
            return None
        return OAuthClientInformationFull.model_validate_json(decrypt_str(doc["client_info_enc"]))

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await mcp_oauth_tokens_collection.update_one(
            self._key(),
            {"$set": {
                "client_info_enc": encrypt_str(client_info.model_dump_json()),
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    async def clear(self) -> None:
        await mcp_oauth_tokens_collection.delete_one(self._key())

    async def has_tokens(self) -> bool:
        doc = await mcp_oauth_tokens_collection.find_one(self._key(), {"tokens_enc": 1})
        return bool(doc and doc.get("tokens_enc"))
