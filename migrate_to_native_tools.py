"""
Migration script: Remove built-in MCP servers and initialize native tools
Run this once to migrate from MCP-based to native tools architecture
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

async def migrate_to_native_tools():
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.chatbot
    mcp_servers_collection = db.mcp_servers
    tools_collection = db.tools
    
    print("="*60)
    print("MIGRATION: MCP Servers -> Native Tools")
    print("="*60)
    
    # Step 1: Remove built-in MCP servers
    print("\nStep 1: Removing built-in MCP servers...")
    result = await mcp_servers_collection.delete_many({
        "server_id": {"$in": ["local_demo_server", "google_drive_mcp"]}
    })
    print(f"  Deleted {result.deleted_count} built-in MCP server(s)")
    
    # Step 2: Create tools collection with unique index
    print("\nStep 2: Creating tools collection...")
    try:
        await tools_collection.create_index("tool_id", unique=True, sparse=True)
        print("  Created unique index on tool_id")
    except Exception as e:
        print(f"  Index might already exist: {e}")
    
    # Step 3: Register native tools
    print("\nStep 3: Registering native tools...")
    
    # Import tools
    from tools import get_all_tools
    
    tools_registered = 0
    for tool in get_all_tools():
        try:
            await tools_collection.update_one(
                {"tool_id": tool.name},
                {"$set": {
                    "tool_id": tool.name,
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "requires_auth": tool.requires_auth,
                    "is_enabled": True,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            tools_registered += 1
            print(f"  [OK] {tool.name}")
        except Exception as e:
            print(f"  [FAIL] {tool.name}: {e}")
    
    print(f"\n  Total tools registered: {tools_registered}")
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    
    mcp_count = await mcp_servers_collection.count_documents({})
    tools_count = await tools_collection.count_documents({})
    
    print(f"MCP Servers remaining: {mcp_count} (user-added only)")
    print(f"Native Tools registered: {tools_count}")
    
    print("\n[SUCCESS] Migration complete!")
    print("\nNext steps:")
    print("1. Restart your backend server")
    print("2. Test native tools in chat")
    print("3. External MCP servers (if any) will still work")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(migrate_to_native_tools())
