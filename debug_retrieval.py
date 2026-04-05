import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

async def debug_retrieval_flow():
    print("\n--- 🕵️ Retrieval Debugger ---")
    
    # 1. Setup
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY missing!")
        return

    print("1. Initializing Embeddings (models/text-embedding-004)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    
    # 2. Embed Query
    query = "internship"
    print(f"2. Embedding query: '{query}'")
    try:
        query_vector = embeddings.embed_query(query)
        print(f"   ✅ Generated Vector (Dim: {len(query_vector)})")
        print(f"   Sample: {query_vector[:5]}...")
    except Exception as e:
        print(f"   ❌ Embedding Failed: {e}")
        return

    # 3. Connect to Qdrant
    print("3. Connecting to Local Qdrant...")
    qdrant_path = "./qdrant_data"
    
    if os.path.exists("./backend/qdrant_data"):
        qdrant_path = "./backend/qdrant_data"
        print(f"   ✅ Found at '{qdrant_path}'")
    elif os.path.exists("./qdrant_data"):
        print(f"   ✅ Found at '{qdrant_path}'")
    else:
        print("   ❌ ./qdrant_data AND ./backend/qdrant_data do not exist!")
        return
        
    client = QdrantClient(path=qdrant_path)
    
    # Debug: Print available attributes
    # print(f"   ℹ️ Client type: {type(client)}")
    # print(f"   ℹ️ Client dir: {dir(client)}")

    collection_name = "agentic_rag_v1"
    
    # Check count
    count = client.count(collection_name).count
    print(f"   📊 Collection '{collection_name}' has {count} points.")
    
    if count == 0:
        print("   ❌ Index is empty. Nothing to search.")
        return

    # 4. Perform Search
    print("4. Executing Raw Vector Search...")
    
    results = []
    try:
        # Try standard search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
            with_payload=True
        )
    except AttributeError:
        print("   ⚠️ client.search not found! Trying query_points...")
        try:
             # Fallback for newer/different client structure
             results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=5,
                with_payload=True
             ).points
        except Exception as e:
            print(f"   ❌ Query Points also failed: {e}")
            return
    except Exception as e:
         print(f"   ❌ Search failed with unexpected error: {e}")
         return
    print(f"   📊 Collection '{collection_name}' has {count} points.")
    
    if count == 0:
        print("   ❌ Index is empty. Nothing to search.")
        return

    print(f"   ✅ Search returned {len(results)} results.")
    
    if len(results) == 0:
        print("   ⚠️ No results found even without threshold!")
    
    for i, res in enumerate(results):
        print(f"\n   [Result {i}] Score: {res.score:.4f}")
        try:
             # Handle both object (payload.get) and dictionary access if different client version
             payload = res.payload
             if hasattr(payload, 'get'):
                source = payload.get('metadata', {}).get('source', 'Unknown')
                content = payload.get('page_content', '')[:100].replace('\n', ' ')
             else:
                source = "Unknown (Payload format diff)"
                content = str(payload)[:100]
                
             print(f"   Source: {source}")
             print(f"   Content: {content}...")
        except Exception:
             print(f"   Payload: {res.payload}")

    print("\n--- End Debug ---")

if __name__ == "__main__":
    asyncio.run(debug_retrieval_flow())
