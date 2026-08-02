import os
from typing import List, Optional, Dict, Any
from qdrant_client import models
from rag.vector_store.qdrant_manager import QdrantManager
from rag.rerank import rerank, RERANK_ENABLED
import logging

# how many candidates to pull before the reranker narrows to `limit`
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

import structlog
logger = structlog.get_logger(__name__)

# Initialize QdrantManager once
qdrant_manager = QdrantManager()

from langchain_core.tools import tool

@tool
def search_knowledge_base(
    query: str, 
    selected_files: Optional[List[str]] = None,
    limit: int = 5,
    offset: int = 0,
    user_id: str = None
) -> List[Dict[str, Any]]:
    """
    Agent Tool: Searches the knowledge base (Vector DB) for relevant context.
    
    Args:
        query (str): The semantic search query.
        selected_files (List[str], optional): List of filenames to restrict search to.
                                              If empty or None, searches all files (unless enforced upstream).
        limit (int): Number of chunks to return (default: 5).
        offset (int): Pagination offset (default: 0).
        user_id (str, optional): The user ID to restrict the search to.
        
    Returns:
        List[Dict]: retrieved chunks. Each has:
          - content:  the chunk text
          - source:   the document filename (cite this)
          - section:  the heading breadcrumb, e.g. "Chapter 1 > Socialisation"
                      (cite the section the fact came from when answering)
          - page:     the page number in the source PDF
          - figures:  (only when the chunk contains an image)
                      [{"filename", "url"}] — to reference a figure in your answer,
                      embed it as markdown: ![<filename>](<url>)
          - score:    similarity score (higher = more relevant)

    When you answer from these results, cite the source and section (e.g.
    "(AS sociology.pdf — Socialisation, p.7)"), and if a relevant chunk carries a
    figure, include the image with ![filename](url).
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query provided to search_knowledge_base")
            return [{"error": "You must provide a non-empty query string to search."}]

        # ── TENANCY GUARD ──────────────────────────────────────────────
        # user_id is mandatory. Without it the Qdrant filter would omit the
        # user condition and search EVERY user's documents. Refuse instead of
        # leaking. The authenticated user_id is injected by agent_tool_node.
        if not user_id:
            logger.error("🔒 search_knowledge_base called without user_id — refusing (tenancy guard)")
            return [{"error": "No user context available; the knowledge base cannot be searched."}]

        logger.info(f"🛠️ Tool Call: search_knowledge_base(query='{query}', files={selected_files}, limit={limit}, offset={offset}, user_id={user_id})")
        
        vector_store = qdrant_manager.get_vector_store()
        
        # 1. Build Filter
        search_kwargs = {
            "k": limit,
            "offset": offset, # Note: LangChain wrapper might not support raw 'offset' in as_retriever directly
                              # We might need to use the raw client methods for pagination + filtering
                              # LangChain's similarity_search usually takes k but not offset.
                              # Let's use QdrantClient directly for maximum control over offset.
        }
        
        # Construct Filter by user_id and file_id UUID
        must_conditions = []
        if user_id:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.user_id",
                    match=models.MatchValue(value=user_id)
                )
            )
            
        if selected_files:
             must_conditions.append(
                models.FieldCondition(
                    key="metadata.file_id",   # UUID — not filename
                    match=models.MatchAny(any=selected_files)
                )
            )
            
        qdrant_filter = None
        if must_conditions:
            qdrant_filter = models.Filter(must=must_conditions)

        # Retrieve a WIDER candidate net when reranking, then the CPU cross-encoder
        # narrows it to the best `limit`. This "retrieve wide, rerank narrow" is the
        # biggest precision lever after embeddings.
        fetch_k = max(limit, RERANK_CANDIDATES) if RERANK_ENABLED else limit
        results = vector_store.similarity_search_with_score(
            query=query,
            k=fetch_k,
            filter=qdrant_filter  # LangChain-Qdrant supports passing the models.Filter
        )

        # Format Results
        output = []
        for doc, score in results:
            payload = doc.metadata or {}

            item = {
                "content": doc.page_content,
                "source": payload.get("source", "unknown"),
                "section": payload.get("section_path") or payload.get("section"),
                "page": payload.get("page"),
                "score": score,
                "json_id": payload.get("json_id") or payload.get("json_file_id", "unknown"),
                "document_id": payload.get("_id", "unknown"),
            }
            # figures in this chunk — filename + durable URL so the agent can show them
            figures = payload.get("figures")
            if figures:
                item["figures"] = figures
            output.append(item)

        # Rerank the wide candidate net down to the best `limit` (CPU cross-encoder).
        if RERANK_ENABLED and len(output) > limit:
            before = len(output)
            output = rerank(query, output, text_key="content", top_k=limit)
            logger.info(f"   🎯 Reranked {before} → {len(output)} chunks.")
        else:
            output = output[:limit]

        logger.info(f"   ✅ Found {len(output)} chunks.")
        if not output:
            # Give the agent an explicit off-ramp instead of a bare `[]` — without
            # this, a zero-result turn tends to spiral into repeated query
            # rewordings (and even sandbox exploration hunting for the "document")
            # because the model has no signal that zero really means zero.
            return [{
                "info": (
                    "No matching chunks found for this query. If a couple of "
                    "reasonably different phrasings all return nothing, the user "
                    "most likely has no matching document ingested in their "
                    "knowledge base yet — say so plainly and ask them to upload "
                    "it, rather than continuing to retry variations or looking "
                    "for it in the sandbox (search_knowledge_base and the sandbox "
                    "are separate; a knowledge-base document is never a sandbox file)."
                )
            }]
        return output

    except Exception as e:
        logger.error(f"❌ Search Tool Failed: {e}")
        return []
