from qdrant_client import QdrantClient
import os
import shutil

def reset_qdrant():
    print("🗑️ Resetting Qdrant Data...")
    
    # 1. Try to delete via Client
    try:
        if os.path.exists("./qdrant_data"):
             client = QdrantClient(path="./qdrant_data")
             collection_name = "agentic_rag_v1"
             client.delete_collection(collection_name)
             print(f"✅ Collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"⚠️ Failed to delete collection via client: {e}")

    # 2. Hard Delete Folder (Nuclear Option)
    # This is safe because we are using Local persistence
    # But files might be locked if Uvicorn is running.
    # So we advise user to stop server first.
    print("ℹ️ For a full reset, ensure the server is STOPPED.")
    
    # We won't delete the folder programmatically because of Windows file locks.
    # The delete_collection above is usually sufficient.
    
    print("✨ Reset Complete. Please restart Uvicorn.")

if __name__ == "__main__":
    reset_qdrant()
