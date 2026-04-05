import os
import asyncio
from dotenv import load_dotenv
from rag.vector_store.qdrant_manager import QdrantManager
from langchain_core.documents import Document

load_dotenv()

async def debug_langchain_flow():
    print("\n--- 🦜🔗 LangChain Retrieval Debugger ---")
    
    # 1. Initialize Manager
    print("1. Initializing QdrantManager...")
    try:
        manager = QdrantManager()
        vector_store = manager.get_vector_store()
        print("   ✅ Manager Initialized")
    except Exception as e:
        print(f"   ❌ Manager Initialization Failed: {e}")
        return

    # 2. Check Client Path
    print(f"   ℹ️ Client Info: {manager.client}")
    # Hack to verify collection count via client
    col_name = manager.collection_name
    try:
        count = manager.client.count(col_name).count
        print(f"   📊 Collection '{col_name}' has {count} points (Verified via Client)")
    except Exception as e:
        print(f"   ⚠️ Could not verify count: {e}")

    # 3. Setup Retriever
    print("3. Setting up Retriever (Similarity)...")
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    except Exception as e:
        print(f"   ❌ Retriever Setup Failed: {e}")
        return
        
    # 4. Invoke
    query = "internship"
    print(f"4. Invoking Retriever with query: '{query}'...")
    try:
        docs = retriever.invoke(query)
        print(f"   ✅ Invoke returned {len(docs)} documents.")
        
        for i, doc in enumerate(docs):
            print(f"\n   [Doc {i}] Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Retrieval Invoke Failed: {e}")

    print("\n--- End Debug ---")

if __name__ == "__main__":
    asyncio.run(debug_langchain_flow())
