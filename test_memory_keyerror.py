import asyncio
from services.memory_service import MemoryService
from core.database import db

async def test():
    user_id = "6a1c178b7a3b95171815a556"
    # insert a mock memory without topic
    db["user_memories"].insert_one({"user_id": user_id, "memories": [{"content": "no topic here"}]})
    
    await MemoryService.extract_and_store(
        user_id=user_id,
        human_message="my name is ankit",
        ai_response="Hello Ankit! It's great to meet you."
    )

if __name__ == "__main__":
    asyncio.run(test())
