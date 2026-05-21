import os
import logging
import pathlib
from qdrant_client import QdrantClient, models
try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages the Qdrant Vector DB connection and collection setup.
    Supports:
    - Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY set) → production
    - Local Server (QDRANT_URL set, no key)           → dev with docker
    - Embedded Local Mode (fallback)                  → dev without docker
    Singleton pattern to prevent multiple connections.
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

        # --- Connection Strategy ---
        if self.api_key:
            # CLOUD MODE: API key provided — must be Qdrant Cloud
            logger.info(f"☁️  Connecting to Qdrant Cloud: {self.url}")
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=20,
            )
            # Verify cloud connection immediately on startup
            try:
                self.client.get_collections()
                logger.info("✅ Qdrant Cloud connected successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Qdrant Cloud: {e}")
                raise RuntimeError(
                    f"Could not reach Qdrant Cloud at {self.url}. "
                    "Check QDRANT_URL and QDRANT_API_KEY."
                ) from e
        else:
            # LOCAL MODE: Try local server first, then embedded fallback
            try:
                logger.info(f"🔌 Connecting to local Qdrant server: {self.url}")
                self.client = QdrantClient(url=self.url, timeout=5)
                self.client.get_collections()
                logger.info("✅ Local Qdrant server connected.")
            except Exception:
                backend_root = pathlib.Path(__file__).parent.parent.parent
                qdrant_path = backend_root / "qdrant_data"
                self._remove_stale_lock(qdrant_path)
                logger.warning(
                    f"⚠️  Local server unreachable. Using EMBEDDED mode at {qdrant_path}"
                )
                self.client = QdrantClient(path=str(qdrant_path))

        self._initialized = True

    def _remove_stale_lock(self, db_path):
        lock_file = pathlib.Path(db_path) / ".lock"
        if lock_file.exists():
            try:
                os.remove(lock_file)
                logger.warning(f"🔓 Removed stale lock: {lock_file}")
            except Exception as e:
                logger.error(f"Failed to remove lock: {e}")

    def ensure_collection(self):
        """
        Idempotent: Creates the collection if it doesn't exist.
        Dimension: 768 (text-embedding-004), Distance: Cosine
        """
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)

            if not exists:
                logger.info(f"📦 Creating collection '{self.collection_name}' (dim=768, cosine)")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"✅ Collection '{self.collection_name}' created.")
            else:
                logger.info(f"✅ Collection '{self.collection_name}' already exists.")

        except Exception as e:
            logger.error(f"❌ ensure_collection failed: {e}")
            raise

    def list_unique_sources(self) -> list:
        """
        Scans up to 1000 points and returns unique source filenames.
        """
        try:
            self.ensure_collection()
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["source", "metadata"],
                with_vectors=False
            )
            sources = set()
            for point in points:
                if not point.payload:
                    continue
                # LangChain stores metadata nested under "metadata" key
                if "metadata" in point.payload and isinstance(point.payload["metadata"], dict):
                    src = point.payload["metadata"].get("source")
                    if src:
                        sources.add(src)
                elif "source" in point.payload:
                    sources.add(point.payload["source"])

            logger.info(f"📂 Found {len(sources)} unique sources in Qdrant.")
            return list(sources)

        except Exception as e:
            logger.error(f"Failed to list sources: {e}")
            return []

    def get_vector_store(self) -> QdrantVectorStore:
        """Returns the LangChain QdrantVectorStore wrapper."""
        self.ensure_collection()
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
