"""
Database cleanup script to remove duplicate MCP servers
Run this once to clean up the database
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def cleanup_mcp_servers():
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.chatbot
    mcp_servers_collection = db.mcp_servers
    
    print("Cleaning up duplicate MCP servers...")
    
    # Delete all old entries without server_id
    result = await mcp_servers_collection.delete_many({"server_id": {"$exists": False}})
    print(f"Deleted {result.deleted_count} old MCP server entries without server_id")
    
    # Keep only one entry per server_id
    server_ids = ["local_demo_server", "google_drive_mcp"]
    
    for server_id in server_ids:
        # Find all entries with this server_id
        entries = await mcp_servers_collection.find({"server_id": server_id}).sort("created_at", -1).to_list(length=100)
        
        if len(entries) > 1:
            print(f"Found {len(entries)} entries for {server_id}, keeping only the latest one")
            # Keep the first (latest), delete the rest
            ids_to_delete = [entry["_id"] for entry in entries[1:]]
            result = await mcp_servers_collection.delete_many({"_id": {"$in": ids_to_delete}})
            print(f"Deleted {result.deleted_count} duplicate entries for {server_id}")
        elif len(entries) == 1:
            print(f"OK: Only 1 entry found for {server_id}")
        else:
            print(f"WARNING: No entries found for {server_id}")
    
    # Count remaining entries
    total = await mcp_servers_collection.count_documents({})
    print(f"\nFinal count: {total} MCP server(s) in database")
    print("Database cleanup complete!")
    client.close()

if __name__ == "__main__":
    asyncio.run(cleanup_mcp_servers())
