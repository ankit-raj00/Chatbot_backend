from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

def debug_qdrant():
    print("--- 🕵️ Qdrant Debugger ---")
    
    # Connect to Local Qdrant
    if os.path.exists("./qdrant_data"):
        print(f"✅ Found local qdrant_data directory.")
        client = QdrantClient(path="./qdrant_data")
    else:
        print("❌ No local qdrant_data directory found!")
        return

    collection_name = "agentic_rag_v1"
    
    # 1. Check Collections
    collections = client.get_collections()
    print(f"Collections found: {[c.name for c in collections.collections]}")
    
    if collection_name not in [c.name for c in collections.collections]:
        print(f"❌ Collection '{collection_name}' DOES NOT EXIST.")
        print("Creating it now just to be sure...")
        # client.create_collection(...) # Don't create, we want to see why it's missing
        return

    # 2. Check Points
    count = client.count(collection_name)
    print(f"📊 Total Points in '{collection_name}': {count.count}")
    
    if count.count == 0:
        print("⚠️ Collection is empty. Ingestion failed to save vectors.")
        return

    # 3. Inspect Payload
    print("🔍 Inspecting first 3 points...")
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    
    for i, point in enumerate(points):
        print(f"\n[Point {i}] ID: {point.id}")
        print(f"Payload: {point.payload}")
        
    print("\n--- End Debug ---")

if __name__ == "__main__":
    debug_qdrant()
