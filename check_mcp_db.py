"""
Check what's actually in the database
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_database():
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.chatbot
    mcp_servers_collection = db.mcp_servers
    
    print("=== Current MCP Servers in Database ===\n")
    
    # Get all servers
    servers = await mcp_servers_collection.find({}).to_list(length=100)
    
    if not servers:
        print("No servers found in database")
    else:
        for i, server in enumerate(servers, 1):
            print(f"Server #{i}:")
            print(f"  _id: {server.get('_id')}")
            print(f"  server_id: {server.get('server_id', 'NOT SET')}")
            print(f"  name: {server.get('name')}")
            print(f"  url: {server.get('url')}")
            print(f"  user_id: {server.get('user_id', 'NOT SET')}")
            print(f"  is_local: {server.get('is_local', 'NOT SET')}")
            print()
    
    print(f"Total: {len(servers)} server(s)")
    client.close()

if __name__ == "__main__":
    asyncio.run(check_database())
