import os
import sys

# ── Configure structured logging FIRST ──────────────────────
# Must be called before any logging.getLogger() calls in other modules
from core.logging_config import configure_logging
configure_logging()
from core.request_context import CorrelationIdMiddleware
# ────────────────────────────────────────────────────────────

# stdout/stderr redirection removed for console visibility

print(f"Executable: {sys.executable}")
print(f"Path: {sys.path}")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import routes
from routes.auth_routes import router as auth_router
from routes.conversation_routes import router as conversation_router
from routes.chat_routes import router as chat_router
# from routes.mcp_routes import router as mcp_router # Legacy
from routes.mcp_server_routes import router as mcp_server_router
from routes.oauth_routes import router as oauth_router
from routes.tool_routes import router as tool_router
from routes.auth_status_routes import router as auth_status_router
from routes.user_routes import router as user_router
from routes.upload_routes import router as upload_router
from routes.rag_routes import router as rag_router
from routes.admin_routes import router as admin_router
# ── Phase 2b / Phase 8 new routes ────────────────────────────
from routes.skill_vault_routes import router as skill_vault_router
from routes.agent_routes import router as agent_router
from routes.output_routes import router as output_router

from contextlib import asynccontextmanager

from core.cache import init_redis, close_redis
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
from core.limiter import limiter

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for initialization and cleanup"""

    # ── Phase 2: Initialize Redis ─────────────────────────────
    try:
        await init_redis()
    except RuntimeError as e:
        print(f"⚠️  Redis startup warning: {e}")
        print("    App will continue but caching features will be disabled.")
        # Do NOT raise here in dev — allow app to start without Redis
        # In production, set REDIS_URL correctly so this never triggers

    async def ensure_indexes():
        """Create MongoDB indexes for query performance."""
        from core.database import (
            messages_collection, conversations_collection,
            users_collection
        )
        # messages: fetch by conversation_id + user_id (most common query)
        await messages_collection.create_index(
            [("conversation_id", 1), ("user_id", 1), ("timestamp", 1)]
        )
        # conversations: list by user, sorted by updated_at
        await conversations_collection.create_index(
            [("user_id", 1), ("updated_at", -1)]
        )
        # users: lookup by email
        await users_collection.create_index("email", unique=True)

    try:
        await ensure_indexes()
        print("✅ MongoDB indexes verified")
    except Exception as e:
        print(f"⚠️  Index creation warning: {e}")

    # Startup: Initialize native tools
    try:
        from core.database import tools_collection
        from tools import get_all_tools
        from datetime import datetime
        
        print("="*60)
        print("INITIALIZING NATIVE TOOLS")
        print("="*60)
        
        # Create unique index on tool_id
        try:
            await tools_collection.create_index("tool_id", unique=True, sparse=True)
            print("✅ Created unique index on tool_id")
        except Exception as e:
            print(f"ℹ️  Index already exists: {e}")
        
        # Register all native tools
        tools_registered = 0
        for tool in get_all_tools():
            try:
                await tools_collection.update_one(
                    {"tool_id": tool.name},
                    {"$set": {
                        "tool_id": tool.name,
                        "name": tool.name,
                        "description": tool.description,
                        "category": tool.metadata.get("category", "general"),
                        "requires_auth": tool.metadata.get("requires_auth", False),
                        "is_enabled": True,
                        "updated_at": datetime.now()
                    }},
                    upsert=True
                )
                tools_registered += 1
                print(f"  ✓ {tool.name}")
            except Exception as e:
                print(f"  ✗ Failed to register {tool.name}: {e}")
        
        print(f"\n✅ Registered {tools_registered} native tools")
        print("="*60)
        
    except Exception as e:
        print(f"⚠️  Startup error: {e}")
        # Don't fail startup if tools registration fails
        pass

    # ── LangSmith tracing ─────────────────────────────────────
    # If LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set,
    # LangSmith will automatically trace all LangGraph runs.
    # No code changes needed — the langsmith package hooks in automatically.
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    if langsmith_enabled:
        print(f"✅ LangSmith tracing enabled — project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print("ℹ️  LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")

    # ── Supervisor graph init ──────────────────────────────────
    try:
        from graph.supervisor import get_supervisor
        await get_supervisor()   # warm up — creates Redis checkpointer connection
        print("✅ Supervisor graph initialized")
    except Exception as e:
        print(f"⚠️  Supervisor init warning: {e}")

    # ── Workspace cleanup background task ─────────────────────
    import asyncio
    try:
        from utils.workspace_cleanup import run_cleanup_loop
        asyncio.create_task(run_cleanup_loop())
        print("✅ Workspace cleanup task started")
    except Exception as e:
        print(f"⚠️  Cleanup task warning: {e}")

    yield  # App is running

    # Shutdown: Cleanup MCP connections
    print("🧹 Shutting down: Cleaning up MCP connections")
    from utils.mcp_connection_manager import mcp_manager
    await mcp_manager.disconnect_all()

    # Shutdown: close supervisor checkpointer
    try:
        from graph.supervisor import close_supervisor
        await close_supervisor()
        print("✅ Supervisor checkpointer closed")
    except Exception:
        pass

    # Phase 2: Close Redis
    await close_redis()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Gemini MCP Chat API",
    description="Complete authentication system with Gemini + MCP integration",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a clean 422 with field-level error details."""
    import structlog
    log = structlog.get_logger("validation")
    log.warning("request.validation_error", path=str(request.url.path), errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — logs the exception and returns a clean 500."""
    import structlog
    log = structlog.get_logger("global_error")
    log.error(
        "unhandled.exception",
        path=str(request.url.path),
        error=str(exc),
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
        }
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add CORS middleware
_allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173"   # dev fallback
)
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],   # Explicit, not ["*"]
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

# ── Pure ASGI logging middleware (avoids BaseHTTPMiddleware anyio conflicts) ──
import structlog as _structlog
import time as _time

_access_log = _structlog.get_logger("access")

class LoggingMiddleware:
    """Structured access logging middleware."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = _time.monotonic()
        request = Request(scope, receive)
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (_time.monotonic() - start) * 1000
            _access_log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(elapsed_ms, 2),
            )

# Include routers
app.include_router(auth_router)
app.include_router(conversation_router)
app.include_router(chat_router)
# app.include_router(mcp_router) # Legacy
app.include_router(mcp_server_router)
app.include_router(oauth_router)
app.include_router(tool_router)
app.include_router(auth_status_router)
app.include_router(user_router)
app.include_router(upload_router)
app.include_router(rag_router)
app.include_router(admin_router)
# ── Phase 2b / Phase 8 new routers ──────────────────────────
app.include_router(skill_vault_router)
app.include_router(agent_router)
app.include_router(output_router)

# Attach middlewares
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(LoggingMiddleware)  # type: ignore[arg-type]

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Root endpoint"""
    return {
        "message": "Gemini MCP Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Deep health check — verifies all critical dependencies.
    Returns 200 if healthy, 503 if any dependency is down.
    Used by Render for health check URL.
    """
    from core.database import users_collection
    from core.cache import get_redis

    checks = {}
    all_ok = True

    # MongoDB check
    try:
        await users_collection.find_one({}, {"_id": 1})
        checks["mongodb"] = "ok"
    except Exception as e:
        checks["mongodb"] = f"error: {str(e)[:50]}"
        all_ok = False

    # Redis check
    try:
        r = await get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)[:50]}"
        # Redis failure is non-fatal — app still works, just slower

    # Qdrant check (lightweight)
    try:
        from rag.vector_store.qdrant_manager import QdrantManager
        mgr = QdrantManager()
        mgr.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)[:50]}"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
            "version": os.getenv("SERVICE_VERSION", "unknown")
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
