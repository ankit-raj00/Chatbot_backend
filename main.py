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
from routes.mcp_routes import router as mcp_router
from routes.mcp_server_routes import router as mcp_server_router
from routes.oauth_routes import router as oauth_router

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
        "https://chatbot-backend-beta-nine.vercel.app"
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

# Startup event to initialize local MCP server
@app.on_event("startup")
async def startup_event():
    # --- Safe Startup Wrapper ---
    try:
        from core.database import mcp_servers_collection
        from utils.mcp_connection_manager import mcp_manager
        import pathlib
        
        # Get local MCP server path
        local_mcp_path = str(pathlib.Path(__file__).parent / "services" / "mcp_server.py")
        
        # Check if local MCP server exists in database
        existing = await mcp_servers_collection.find_one({"url": local_mcp_path})
        
        if not existing:
            # Add local MCP server to database
            local_server = {
                "name": "Local Demo Server",
                "url": local_mcp_path,
                "description": "Local MCP server with demo tools (roll_dice, get_weather)",
                "is_local": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            await mcp_servers_collection.insert_one(local_server)
            print(f"✅ Added local MCP server to database: {local_mcp_path}")
        
        # Pre-connect to local MCP server
        print(f"🔌 Pre-connecting to local MCP server...")
        # In Vercel, subprocess execution might fail, so we catch it
        try:
            await mcp_manager.connect(local_mcp_path)
            print(f"✅ Local MCP server ready")
        except Exception as e:
            print(f"⚠️ Failed to connect to local MCP server (expected in Vercel): {e}")

        # --- Google Drive MCP Server ---
        drive_mcp_path = str(pathlib.Path(__file__).parent / "services" / "google_drive_server.py")
        
        # Check if Drive MCP server exists
        existing_drive = await mcp_servers_collection.find_one({"url": drive_mcp_path})
        
        if not existing_drive:
            # Add Drive MCP server to database
            drive_server = {
                "name": "Google Drive MCP",
                "url": drive_mcp_path,
                "description": "Google Drive integration (List folders, Create folders)",
                "is_local": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            await mcp_servers_collection.insert_one(drive_server)
            print(f"✅ Added Google Drive MCP server to database: {drive_mcp_path}")
        
        # Pre-connect to Drive MCP server
        print(f"🔌 Pre-connecting to Google Drive MCP server...")
        try:
            await mcp_manager.connect(drive_mcp_path)
            print(f"✅ Google Drive MCP server ready")
        except Exception as e:
            print(f"⚠️ Failed to connect to Drive MCP server (expected in Vercel): {e}")
            
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
app.include_router(mcp_router)
app.include_router(mcp_server_router)
app.include_router(oauth_router)

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
