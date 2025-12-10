"""
Fix duplicate MCP servers by:
1. Cleaning up existing duplicates
2. Creating a unique index on server_id to prevent future duplicates
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def fix_mcp_duplicates():
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.chatbot
    mcp_servers_collection = db.mcp_servers
    
    print("Step 1: Cleaning up duplicate MCP servers...")
    
    # Keep only one entry per server_id (the latest one)
    server_ids = ["local_demo_server", "google_drive_mcp"]
    
    for server_id in server_ids:
        # Find all entries with this server_id, sorted by created_at (newest first)
        entries = await mcp_servers_collection.find({"server_id": server_id}).sort("created_at", -1).to_list(length=100)
        
        if len(entries) > 1:
            print(f"  Found {len(entries)} entries for '{server_id}', keeping only the latest")
            # Keep the first (latest), delete the rest
            ids_to_delete = [entry["_id"] for entry in entries[1:]]
            result = await mcp_servers_collection.delete_many({"_id": {"$in": ids_to_delete}})
            print(f"  Deleted {result.deleted_count} duplicate(s)")
        elif len(entries) == 1:
            print(f"  OK: Only 1 entry for '{server_id}'")
        else:
            print(f"  No entries found for '{server_id}'")
    
    print("\nStep 2: Creating unique index on server_id...")
    try:
        # Create unique index to prevent future duplicates
        await mcp_servers_collection.create_index("server_id", unique=True, sparse=True)
        print("  Unique index created successfully!")
    except Exception as e:
        print(f"  Index might already exist: {e}")
    
    # Count remaining entries
    total = await mcp_servers_collection.count_documents({})
    print(f"\nFinal count: {total} MCP server(s) in database")
    print("Fix complete! The unique index will prevent duplicates in the future.")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(fix_mcp_duplicates())
