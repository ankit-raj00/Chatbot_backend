import os
import logging
from typing import Optional
from qdrant_client import QdrantClient, models
from qdrant_client import QdrantClient, models
try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    # Fallback/Migration note: older langchain might use different path
    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages the Qdrant Vector DB connection and collection setup.
    Implements Section 3.1: Vector DB with Hybrid Search capabilities.
    Singleton pattern applied to prevent file locking in local mode.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "agentic_rag_v1")
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Client
        if self.api_key:
             self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            # Try to connect to server, fallback to local if fails
            try:
                self.client = QdrantClient(url=self.url)
                self.client.get_collections() # Test connection
            except Exception:
                # Use absolute path to ensure we hit the same DB regardless of CWD
                import pathlib
                current_file = pathlib.Path(__file__)
                # Go up 3 levels: vector_store -> rag -> backend
                backend_root = current_file.parent.parent.parent
                qdrant_path = backend_root / "qdrant_data"
                
                self._remove_stale_lock(qdrant_path)
                
                logger.warning(f"Could not connect to Qdrant at {self.url}. Fallback to LOCAL EMBEDDED mode at {qdrant_path}")
                self.client = QdrantClient(path=str(qdrant_path))
        
        self._initialized = True
        
    def _remove_stale_lock(self, db_path: str):
        """
        Hack to fix 'Embedded Mode' file locking issues on restart.
        Removes the .lock file if it exists.
        """
        import pathlib
        lock_file = pathlib.Path(db_path) / ".lock"
        if lock_file.exists():
            try:
                os.remove(lock_file)
                logger.warning(f"🔓 Forced removal of stale Qdrant lock file at: {lock_file}")
            except Exception as e:
                logger.error(f"Failed to remove lock file: {str(e)}")
             
    def ensure_collection(self):
        """
        Idempotent check to ensure the collection exists with correct config.
        Metric: Cosine
        Dimension: 768 (text-embedding-004)
        """
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                logger.info(f"Creating collection '{self.collection_name}' with dim=768")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768, 
                        distance=models.Distance.COSINE
                    )
                    # Note: For Hybrid Search (Sparse), we would add sparse_vector_config here.
                    # For V1 Phase 1, we start with Dense.
                )
            else:
                logger.info(f"Collection '{self.collection_name}' exists.")
                
        except Exception as e:
            logger.error(f"Failed to check/create Qdrant collection: {str(e)}")
            raise e

    def list_unique_sources(self) -> list[str]:
        """
        List unique 'source' metadata values from the collection.
        Uses Scroll API (Pagination) to scan.
        Note: Qdrant doesn't have a 'SELECT DISTINCT' yet, so this scans.
        For production with millions of points, use a separate 'file_metadata' collection or Redis.
        """
        try:
            self.ensure_collection()
            
            # Limit scan to 1000 points to keep it fast for V1 demo
            # We select ONLY the "source" payload field to reduce bandwidth
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000, 
                with_payload=["source", "metadata"],
                with_vectors=False
            )
            
            sources = set()
            for point in points:
                if point.payload:
                    # Debug: Log first payload to check structure
                    if len(sources) == 0: 
                        logger.info(f"DEBUG Payload Sample: {point.payload}")
                        
                    # LangChain Qdrant wrapper usually stores metadata in "metadata" field within payload
                    # OR flattened depending on config. Let's check both.
                    
                    # Case 1: Flattened (Directly in payload)
                    if "source" in point.payload:
                        sources.add(point.payload["source"])
                        
                    # Case 2: Nested in 'metadata'
                    elif "metadata" in point.payload and isinstance(point.payload["metadata"], dict):
                         if "source" in point.payload["metadata"]:
                             sources.add(point.payload["metadata"]["source"])
            
            return list(sources)
        except Exception as e:
            logger.error(f"Failed to list sources: {e}")
            return []

    def get_vector_store(self) -> QdrantVectorStore:
        """
        Returns the LangChain Qdrant wrapper for use in the graph.
        """
        self.ensure_collection()
        
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model, # Note: param name might be 'embedding' in newer version
        )
