import asyncio
from core.database import messages_collection, conversations_collection, users_collection

async def ensure_indexes():
    print("Creating index for messages...")
    await messages_collection.create_index([("conversation_id", 1), ("user_id", 1), ("timestamp", 1)])
    print("Creating index for conversations...")
    await conversations_collection.create_index([("user_id", 1), ("updated_at", -1)])
    print("Creating index for users...")
    await users_collection.create_index("email", unique=True)
    print("Indexes created.")

if __name__ == "__main__":
    asyncio.run(ensure_indexes())
