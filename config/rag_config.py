"""
Central RAG configuration.
Change RAG_MODEL here and it applies everywhere — no need to edit individual node files.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
# Used by: GraderNode, AgentNode, GenerationNode, HallucinationNode, LlamaParseClient
RAG_MODEL = "gemini-2.5-flash"

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_K = 5          # number of chunks to retrieve per query
CHUNK_PREVIEW = 500      # max chars per chunk in grader prompt (saves tokens)
