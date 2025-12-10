from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://raj12367633:380axF9NzjbW8DaG@cluster0.ely96.mongodb.net/")
client = AsyncIOMotorClient(MONGO_URI)
db = client.gemini_mcp_chat

# Collections
users_collection = db["users"]
conversations_collection = db["conversations"]
messages_collection = db["messages"]
mcp_servers_collection = db["mcp_servers"]
oauth_tokens_collection = db["oauth_tokens"]
tools_collection = db["tools"]
