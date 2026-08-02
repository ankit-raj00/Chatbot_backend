"""
MCP connection manager — builds real, spec-compliant connections (stdio / legacy
SSE / Streamable HTTP) from a user's stored MCPServer config, using
langchain-mcp-adapters' MultiServerMCPClient (which itself opens a fresh MCP
session per tool call — no manual pooling/reconnect logic needed here).

Scoped per (user_id, server_id) — NOT a single global pool. This matters: an
earlier global-singleton-by-URL design meant one user's connected server (and
its tools, including anything behind an API key or OAuth token) was callable by
every other concurrent user's agent turn. Every public method here takes
user_id explicitly and keys all state by (user_id, server_id).
"""
import asyncio
import os
import time
from typing import Any

import structlog
from bson import ObjectId
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, create_session

from core.database import mcp_servers_collection
from utils.crypto import decrypt_dict_values

logger = structlog.get_logger(__name__)

DEFAULT_HTTP_TIMEOUT = 15.0
DEFAULT_SSE_READ_TIMEOUT = 120.0
TOOL_CACHE_TTL = 300  # 5 minutes


def _backend_base_url() -> str:
    return os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _mcp_oauth_redirect_uri() -> str:
    return f"{_backend_base_url()}/oauth/mcp/callback"


class MCPConnectionManager:
    """Per-user MCP connection + tool-cache manager. Process-wide singleton, but
    every piece of state inside it is keyed by (user_id, server_id)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # (user_id, server_id) -> MultiServerMCPClient
        self._clients: dict[tuple[str, str], MultiServerMCPClient] = {}
        # (user_id, server_id) -> a hashable fingerprint of the config last used
        # to build the client, so an edited server rebuilds instead of staying stale.
        self._fingerprints: dict[tuple[str, str], tuple] = {}
        # (user_id, server_id) -> (tools, cached_at)
        self._tool_cache: dict[tuple[str, str], tuple[list[BaseTool], float]] = {}
        self._initialized = True

    # ── Connection building ──────────────────────────────────────────────

    def _fingerprint(self, server: dict) -> tuple:
        """A hashable snapshot of the fields that affect the actual connection,
        so we can detect "this server's config changed, rebuild the client"."""
        return (
            server.get("transport"),
            server.get("url"),
            server.get("command"),
            tuple(server.get("args") or []),
            tuple(sorted((server.get("env") or {}).items())),
            tuple(sorted((server.get("headers") or {}).items())),
            server.get("auth_type"),
            server.get("oauth_authorized"),
        )

    def _build_connection(self, user_id: str, server: dict) -> Connection:
        transport = server.get("transport", "http")
        server_id = str(server["_id"])

        if transport == "stdio":
            return {
                "transport": "stdio",
                "command": server["command"],
                "args": server.get("args") or [],
                "env": decrypt_dict_values(server.get("env")),
            }

        url = server["url"]
        auth_type = server.get("auth_type", "none")
        conn: dict[str, Any] = {
            "transport": transport,  # "sse" or "http"
            "url": url,
            "timeout": DEFAULT_HTTP_TIMEOUT,
            "sse_read_timeout": DEFAULT_SSE_READ_TIMEOUT,
        }

        if auth_type == "headers":
            headers = decrypt_dict_values(server.get("headers"))
            if headers:
                conn["headers"] = headers
        elif auth_type == "oauth":
            if not server.get("oauth_authorized"):
                raise RuntimeError(
                    f"MCP server '{server.get('name', server_id)}' has not been "
                    "authorized yet. Connect it via Settings -> MCP Servers -> "
                    "Authorize first."
                )
            from utils.mcp_oauth_flow import build_runtime_auth_provider
            conn["auth"] = build_runtime_auth_provider(
                user_id, server_id, url, _mcp_oauth_redirect_uri()
            )

        return conn  # type: ignore[return-value]

    async def _get_or_build_client(self, user_id: str, server: dict) -> MultiServerMCPClient:
        server_id = str(server["_id"])
        key = (user_id, server_id)
        fp = self._fingerprint(server)

        if key in self._clients and self._fingerprints.get(key) == fp:
            return self._clients[key]

        conn = self._build_connection(user_id, server)
        client = MultiServerMCPClient({server_id: conn})
        self._clients[key] = client
        self._fingerprints[key] = fp
        self._tool_cache.pop(key, None)
        logger.info("mcp.client_built", user_id=user_id, server_id=server_id, transport=server.get("transport"))
        return client

    # ── Public API ────────────────────────────────────────────────────────

    async def get_tools_for_servers(self, user_id: str, server_ids: list[str]) -> list[BaseTool]:
        """Fetch (cached, 5min TTL) LangChain tools for exactly the given
        servers, scoped to this user. Ownership-checked: a server must belong
        to user_id or be marked is_local (shared system server)."""
        if not server_ids:
            return []

        try:
            object_ids = [ObjectId(sid) for sid in server_ids]
        except Exception:
            logger.warning("mcp.invalid_server_id", server_ids=server_ids)
            return []

        cursor = mcp_servers_collection.find({
            "_id": {"$in": object_ids},
            "is_active": True,
            "$or": [{"user_id": user_id}, {"is_local": True}],
        })
        servers = await cursor.to_list(length=len(object_ids))

        results = await asyncio.gather(
            *(self._get_server_tools_cached(user_id, s) for s in servers),
            return_exceptions=True,
        )

        per_server_tools: dict[str, list[BaseTool]] = {}
        for server, res in zip(servers, results):
            server_id = str(server["_id"])
            if isinstance(res, Exception):
                logger.warning("mcp.get_tools_failed", user_id=user_id, server_id=server_id, error=str(res))
                continue
            per_server_tools[server_id] = res

        return self._merge_tools_collision_safe(servers, per_server_tools)

    async def _get_server_tools_cached(self, user_id: str, server: dict) -> list[BaseTool]:
        server_id = str(server["_id"])
        key = (user_id, server_id)
        now = time.monotonic()

        cached_tools, cached_at = self._tool_cache.get(key, (None, 0.0))
        if cached_tools is not None and (now - cached_at) < TOOL_CACHE_TTL:
            return cached_tools

        client = await self._get_or_build_client(user_id, server)
        try:
            tools = await asyncio.wait_for(client.get_tools(), timeout=DEFAULT_HTTP_TIMEOUT + 5)
            self._tool_cache[key] = (tools, now)
            return tools
        except Exception:
            if cached_tools is not None:
                logger.warning("mcp.tools_stale_fallback", user_id=user_id, server_id=server_id)
                return cached_tools
            raise

    def _merge_tools_collision_safe(
        self, servers: list[dict], per_server_tools: dict[str, list[BaseTool]]
    ) -> list[BaseTool]:
        """Merge tool lists from multiple servers; if two servers expose a tool
        with the same name, prefix BOTH with their server name so neither
        silently shadows the other in the agent's tool map."""
        name_owners: dict[str, set[str]] = {}
        names_by_server: dict[str, list[str]] = {}
        for server_id, tools in per_server_tools.items():
            names_by_server[server_id] = [t.name for t in tools]
            for t in tools:
                name_owners.setdefault(t.name, set()).add(server_id)
        colliding_names = {name for name, owners in name_owners.items() if len(owners) > 1}

        server_names = {str(s["_id"]): s.get("name", str(s["_id"])) for s in servers}

        merged: list[BaseTool] = []
        for server_id, tools in per_server_tools.items():
            prefix = "".join(c if c.isalnum() else "_" for c in server_names[server_id])
            for t in tools:
                if t.name in colliding_names:
                    new_name = f"{prefix}_{t.name}"
                    logger.info("mcp.tool_name_collision", tool=t.name, server_id=server_id, renamed_to=new_name)
                    t = t.model_copy(update={"name": new_name})
                merged.append(t)
        return merged

    async def disconnect(self, user_id: str, server_id: str) -> None:
        key = (user_id, server_id)
        self._clients.pop(key, None)
        self._fingerprints.pop(key, None)
        self._tool_cache.pop(key, None)
        logger.info("mcp.disconnected", user_id=user_id, server_id=server_id)

    def invalidate_tool_cache(self, user_id: str, server_id: str) -> None:
        self._tool_cache.pop((user_id, server_id), None)

    async def test_connection(self, user_id: str, server: dict, timeout: float = 15.0) -> dict:
        """Connect fresh (bypassing any cache) and list tools once. Real error
        messages are surfaced — never rewritten into a reassuring guess."""
        try:
            conn = self._build_connection(user_id, server)
        except Exception as e:
            return {"status": "error", "error": str(e), "tools": []}

        try:
            async with create_session(conn) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            tools = [{"name": t.name, "description": t.description or ""} for t in (result.tools or [])]
            return {"status": "connected", "tools": tools}
        except asyncio.TimeoutError:
            return {"status": "error", "error": f"Connection timed out after {timeout}s", "tools": []}
        except Exception as e:
            return {"status": "error", "error": str(e), "tools": []}

    # ── Resources / prompts (used for system-prompt context injection) ─────

    async def get_available_resources(self, user_id: str, server_ids: list[str]) -> list[dict]:
        all_resources = []
        servers = await self._fetch_servers(user_id, server_ids)
        for server in servers:
            server_id = str(server["_id"])
            try:
                client = await self._get_or_build_client(user_id, server)
                async with client.session(server_id) as session:
                    result = await session.list_resources()
                    for r in result.resources or []:
                        all_resources.append({
                            "uri": r.uri, "name": r.name,
                            "description": r.description or "No description provided",
                            "mimeType": r.mimeType or "application/octet-stream",
                            "source_server_id": server_id,
                        })
            except Exception as e:
                logger.warning("mcp.resources_failed", server_id=server_id, error=str(e))
        return all_resources

    async def get_available_prompts(self, user_id: str, server_ids: list[str]) -> list[dict]:
        all_prompts = []
        servers = await self._fetch_servers(user_id, server_ids)
        for server in servers:
            server_id = str(server["_id"])
            try:
                client = await self._get_or_build_client(user_id, server)
                async with client.session(server_id) as session:
                    result = await session.list_prompts()
                    for p in result.prompts or []:
                        all_prompts.append({
                            "name": p.name, "description": p.description,
                            "arguments": [
                                {"name": a.name, "description": a.description, "required": a.required}
                                for a in (p.arguments or [])
                            ],
                            "source_server_id": server_id,
                        })
            except Exception as e:
                logger.warning("mcp.prompts_failed", server_id=server_id, error=str(e))
        return all_prompts

    async def load_resource(self, user_id: str, server_ids: list[str], uri: str) -> str:
        """Read a single MCP resource by URI, searching only the given
        (user-scoped) servers — not every server ever connected process-wide."""
        servers = await self._fetch_servers(user_id, server_ids)
        for server in servers:
            server_id = str(server["_id"])
            try:
                client = await self._get_or_build_client(user_id, server)
                async with client.session(server_id) as session:
                    result = await session.read_resource(uri)
                    if result and result.contents:
                        parts = []
                        for c in result.contents:
                            if getattr(c, "text", None):
                                parts.append(c.text)
                            elif getattr(c, "blob", None):
                                parts.append(f"[Blob: {getattr(c, 'mimeType', 'unknown')}]")
                        return "".join(parts)
            except Exception as e:
                logger.debug("mcp.resource_not_on_server", server_id=server_id, uri=uri, error=str(e))
                continue
        raise ValueError(f"Resource not found on any selected server: {uri}")

    async def _fetch_servers(self, user_id: str, server_ids: list[str]) -> list[dict]:
        if not server_ids:
            return []
        try:
            object_ids = [ObjectId(sid) for sid in server_ids]
        except Exception:
            return []
        cursor = mcp_servers_collection.find({
            "_id": {"$in": object_ids}, "is_active": True,
            "$or": [{"user_id": user_id}, {"is_local": True}],
        })
        return await cursor.to_list(length=len(object_ids))


# Global instance — internally scoped per (user_id, server_id), see class docstring.
mcp_manager = MCPConnectionManager()
