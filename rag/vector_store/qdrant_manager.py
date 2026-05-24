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
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            output_dimensionality=768,  # MRL truncation keeps dim=768, matching Qdrant collection
            max_retries=6,              # Built-in SDK retry for 429 RESOURCE_EXHAUSTED
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
        Idempotent: Creates the collection if it doesn't exist, then
        ensures the payload index on 'metadata.source' exists.

        Qdrant Cloud REQUIRES explicit payload indexes for filtered search.
        Without this index, any search with a filter on metadata.source
        returns: 400 "Index required but not found for metadata.source".
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

            # ── Payload indexes (required for Qdrant Cloud filtered search) ────────
            # create_payload_index is idempotent — safe to call on every startup.
            for field in ("metadata.source", "metadata.file_id", "metadata.user_id"):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    logger.info(f"🔑 Payload index '{field}' ensured.")
                except Exception as idx_err:
                    err_msg = str(idx_err).lower()
                    if "already exists" in err_msg or "index already" in err_msg:
                        logger.info(f"🔑 Payload index '{field}' already exists.")
                    else:
                        logger.warning(f"⚠️  Could not create payload index '{field}': {idx_err}")


        except Exception as e:
            logger.error(f"❌ ensure_collection failed: {e}")
            raise

    def list_unique_sources(self, user_id: str = None) -> list:
        """
        Scans up to 1000 points and returns unique files as:
            [{"file_id": "<uuid>", "filename": "<original name>"}, ...]

        Returns file_id (for filtering) + filename (for display).
        Falls back to filename-only if file_id is not present on older chunks.
        """
        try:
            self.ensure_collection()
            
            scroll_filter = None
            if user_id:
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["metadata"],
                with_vectors=False,
                scroll_filter=scroll_filter
            )
            # Use dict keyed by file_id to deduplicate
            seen: dict = {}   # file_id → filename
            for point in points:
                if not point.payload:
                    continue
                meta = point.payload.get("metadata", {})
                if not isinstance(meta, dict):
                    continue
                file_id  = meta.get("file_id")
                filename = meta.get("source", "unknown")
                if file_id:
                    seen[file_id] = filename        # prefer UUID key
                elif filename and filename not in seen:
                    seen[filename] = filename       # legacy fallback (no file_id)

            files = [
                {"file_id": fid, "filename": fname}
                for fid, fname in seen.items()
            ]
            logger.info(f"📂 Found {len(files)} unique files in Qdrant.")
            return files

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
