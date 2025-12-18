from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")

if not MONGO_URI:
    # Fallback for local development WITHOUT credentials if needed, or raise error
    # Better to raise error to prevent silent failures or unintended local connections
    raise ValueError("Missing MONGO_URI or MONGODB_URL environment variable")

client = AsyncIOMotorClient(MONGO_URI)
db = client.gemini_mcp_chat

# Collections
users_collection = db["users"]
conversations_collection = db["conversations"]
messages_collection = db["messages"]
mcp_servers_collection = db["mcp_servers"]
oauth_tokens_collection = db["oauth_tokens"]
tools_collection = db["tools"]
