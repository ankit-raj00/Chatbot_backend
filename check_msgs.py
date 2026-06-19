import asyncio
from core.database import messages_collection

async def get_msgs():
    cursor = messages_collection.find({}).sort("timestamp", -1).limit(10)
    msgs = await cursor.to_list(10)
    for m in msgs:
        content = m.get("content", "")
        content_str = str(content)[:100].replace("\n", " ")
        print(f"[{m.get('role')}] {content_str}...")
        if "tool_steps" in m:
            for t in m["tool_steps"]:
                print(f"  Tool: {t.get('name')}")

if __name__ == "__main__":
    asyncio.run(get_msgs())
