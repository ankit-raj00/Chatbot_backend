import os
import sys

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

# Create FastAPI app
app = FastAPI(
    title="Gemini MCP Chat API",
    description="Complete authentication system with Gemini + MCP integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://chatbot-backend-beta-nine.vercel.app",
        "https://chatbot-khaki-eta-53.vercel.app"
        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start_time = time.time()
    print(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"Request completed: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.4f}s")
        return response
    except Exception as e:
        print(f"Request failed: {request.method} {request.url} - Error: {str(e)}")
        raise e

# Startup event to initialize native tools
@app.on_event("startup")
async def startup_event():
    """Initialize native tools on startup"""
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

            
    except Exception as e:
        print(f"❌ Critical Error in Startup Event: {e}")
        # We generally don't want to raise here in Vercel, as it crashes the whole lambda
        # Instead we log it and let the app start without MCPs
        pass

# Shutdown event to cleanup MCP connections
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup MCP connections on shutdown"""
    from utils.mcp_connection_manager import mcp_manager
    await mcp_manager.disconnect_all()

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gemini MCP Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
