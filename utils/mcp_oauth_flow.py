"""
OAuth 2.1 authorization flow orchestration for MCP servers.

Reuses the MCP SDK's `OAuthClientProvider` (mcp.client.auth.oauth2) end-to-end for
the actual RFC mechanics — RFC 9728 protected-resource discovery, RFC 8414
authorization-server metadata discovery, RFC 7591 dynamic client registration,
PKCE, and token exchange/refresh. This module only supplies the two integration
points the SDK expects an application to provide: `redirect_handler` (send the
user to the authorization URL) and `callback_handler` (wait for the authorization
code to come back).

Design: `OAuthClientProvider.async_auth_flow` performs the ENTIRE discovery ->
register -> redirect -> callback -> token-exchange cascade inline, driven by a
single real httpx request through the MCP transport (it 401s, which triggers the
cascade). So "starting" an authorization is: open an MCP session authed with a
fresh provider, which immediately 401s and calls redirect_handler with the real
auth URL. We run that as a background task (it must stay alive while a human
completes the third-party login, which can take minutes) and return the auth URL
to the caller as soon as redirect_handler produces it.

Two coordination mechanisms, deliberately different lifetimes:
  - An in-process asyncio.Event for "the auth URL is ready" — short-lived, never
    crosses a process boundary, so plain asyncio is correct and simplest here.
  - Redis for "the authorization code has arrived" — this MUST survive a real
    gap where a human is off in a different browser tab for an indeterminate
    time, and (per this codebase's existing OAuth state, see
    controllers/oauth_controller.py) must survive a dev-reload / multi-instance
    deployment, which an in-memory dict would not.
"""
import asyncio
import secrets
import time
from urllib.parse import urlparse, parse_qs

import structlog
from mcp.client.auth.oauth2 import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from langchain_mcp_adapters.sessions import create_session, StreamableHttpConnection

from core.cache import cache_get, cache_set, cache_delete
from core.database import mcp_servers_collection
from utils.background_tasks import spawn
from utils.mcp_oauth_storage import MongoTokenStorage
from bson import ObjectId

logger = structlog.get_logger(__name__)

_START_TIMEOUT = 20      # seconds to wait for the initial auth URL
_CALLBACK_TIMEOUT = 300  # seconds to wait for the human to complete login
_CODE_TTL = 120          # seconds the delivered code stays in Redis before pickup

# In-process only — see module docstring for why this is safe here.
_pending_events: dict[str, asyncio.Event] = {}
_pending_urls: dict[str, str] = {}


def _client_metadata(redirect_uri: str) -> OAuthClientMetadata:
    return OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        client_name="AgentX",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


def build_runtime_auth_provider(user_id: str, server_id: str, server_url: str, redirect_uri: str) -> OAuthClientProvider:
    """Provider for RUNTIME tool calls (after the user has already authorized).

    Tokens should already be present and silently refreshable via the stored
    refresh_token; the interactive handlers should essentially never fire here.
    If they do (no valid token at all), fail with a clear, actionable error
    instead of hanging a live chat turn waiting on a callback that will never
    come.
    """
    async def _redirect_handler(url: str) -> None:
        raise RuntimeError(
            f"MCP server '{server_id}' requires authorization. "
            "Connect it via Settings -> MCP Servers -> Authorize before using it."
        )

    async def _callback_handler() -> tuple[str, str | None]:
        raise RuntimeError("MCP server is not authorized (no pending interactive flow at runtime).")

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata(redirect_uri),
        storage=MongoTokenStorage(user_id, server_id),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
        timeout=30.0,
    )


async def _drive_provider(
    *, server_id: str, server_url: str, provider: OAuthClientProvider, flow_id: str,
) -> None:
    """Run in the background: open an authed session, which triggers the full
    OAuth cascade inline, and persist the outcome onto the server document."""
    try:
        conn: StreamableHttpConnection = {
            "transport": "http",
            "url": server_url,
            "auth": provider,
        }
        async with create_session(conn) as session:
            await session.initialize()

        await mcp_servers_collection.update_one(
            {"_id": ObjectId(server_id)},
            {"$set": {"oauth_authorized": True, "oauth_last_error": None}},
        )
        logger.info("mcp_oauth.authorized", server_id=server_id)
    except Exception as e:
        logger.warning("mcp_oauth.flow_failed", server_id=server_id, error=str(e))
        await mcp_servers_collection.update_one(
            {"_id": ObjectId(server_id)},
            {"$set": {"oauth_authorized": False, "oauth_last_error": str(e)[:500]}},
        )
    finally:
        _pending_events.pop(flow_id, None)
        _pending_urls.pop(flow_id, None)


async def initiate_authorize(user_id: str, server_id: str, server_url: str, redirect_uri: str) -> dict:
    """Start an OAuth 2.1 authorization flow for a server. Returns the auth URL
    to redirect the user's browser to. The flow continues in the background
    until the callback route delivers the code (or it times out)."""
    flow_id = secrets.token_urlsafe(16)
    event = asyncio.Event()
    _pending_events[flow_id] = event

    async def _redirect_handler(url: str) -> None:
        _pending_urls[flow_id] = url
        parsed_state = parse_qs(urlparse(url).query).get("state", [None])[0]
        if parsed_state:
            # Correlate the third party's callback (which only echoes back
            # `state`) to our flow_id, via Redis so it survives the gap.
            await cache_set(f"mcp_oauth_flow:{parsed_state}", flow_id, ttl_seconds=_CALLBACK_TIMEOUT + 60)
        event.set()

    async def _callback_handler() -> tuple[str, str | None]:
        deadline = time.monotonic() + _CALLBACK_TIMEOUT
        while time.monotonic() < deadline:
            val = await cache_get(f"mcp_oauth_code:{flow_id}")
            if val:
                await cache_delete(f"mcp_oauth_code:{flow_id}")
                return val["code"], val["state"]
            await asyncio.sleep(1.0)
        raise TimeoutError("Timed out waiting for the OAuth authorization callback")

    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata(redirect_uri),
        storage=MongoTokenStorage(user_id, server_id),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
        timeout=float(_CALLBACK_TIMEOUT),
    )

    # Clear any stale error from a previous attempt before starting a new one.
    await mcp_servers_collection.update_one(
        {"_id": ObjectId(server_id)}, {"$set": {"oauth_last_error": None}}
    )

    spawn(
        _drive_provider(server_id=server_id, server_url=server_url, provider=provider, flow_id=flow_id),
        name=f"mcp_oauth_flow:{server_id}",
    )

    try:
        await asyncio.wait_for(event.wait(), timeout=_START_TIMEOUT)
    except asyncio.TimeoutError as e:
        _pending_events.pop(flow_id, None)
        raise RuntimeError(
            "Timed out starting the OAuth flow — the server may not support "
            "OAuth discovery, or may be unreachable."
        ) from e

    return {"oauth_url": _pending_urls.get(flow_id), "flow_id": flow_id}


async def deliver_callback(code: str, state: str) -> bool:
    """Called by the public OAuth callback route. Returns False if the flow is
    unknown/expired (e.g. user waited too long, or server restarted mid-flow)."""
    flow_id = await cache_get(f"mcp_oauth_flow:{state}")
    if not flow_id:
        return False
    await cache_set(f"mcp_oauth_code:{flow_id}", {"code": code, "state": state}, ttl_seconds=_CODE_TTL)
    return True
