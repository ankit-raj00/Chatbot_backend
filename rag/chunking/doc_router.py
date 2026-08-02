"""
Doc-type routing → chunker profile.

The parser already emits typed blocks, so we classify the document from those
blocks (table/figure density, length, structure) + the filename, and pick a
chunking profile suited to the type. A user-supplied document_type always wins.

Why per-type profiles: a resume is one semantic unit (don't split it); a legal
contract must split tightly on clause headings; financial/medical docs are
table-heavy and want larger chunks so a table keeps its surrounding context;
academic prose reads best in medium semantic chunks.
"""
import os
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# doc_type → chunk_blocks(**profile)
CHUNKER_PROFILES: Dict[str, Dict[str, Any]] = {
    "financial":   {"max_chars": 2000, "min_chars": 400, "overlap_chars": 100},
    "medical":     {"max_chars": 2000, "min_chars": 400, "overlap_chars": 100},
    "legal":       {"max_chars": 1024, "min_chars": 200, "overlap_chars": 80},
    "academic":    {"max_chars": 1500, "min_chars": 300, "overlap_chars": 100},
    "textbook":    {"max_chars": 1400, "min_chars": 300, "overlap_chars": 80},
    "resume":      {"max_chars": 8000, "min_chars": 0,   "overlap_chars": 0, "split_on_headings": False},
    "slides":      {"max_chars": 1200, "min_chars": 0,   "overlap_chars": 0},
    "code_manual": {"max_chars": 1400, "min_chars": 200, "overlap_chars": 60},
    "general":     {"max_chars": 1200, "min_chars": 200, "overlap_chars": 0},
}

_KEYWORDS = {
    "financial": ("balance", "p&l", "financial", "statement", "invoice", "receipt",
                  "10-k", "10k", "10-q", "quarterly", "annual report", "revenue", "ebitda"),
    "legal":     ("contract", "agreement", "nda", "terms", "policy", "regulation",
                  "compliance", "clause", "hereby", "whereas", "party of the"),
    "academic":  ("abstract", "journal", "thesis", "dissertation", "arxiv", "et al",
                  "hypothesis", "methodology", "references", "doi"),
    "medical":   ("patient", "clinical", "diagnosis", "lab", "blood", "mg/dl", "prescription"),
    # NOTE: keep resume terms SPECIFIC — generic words like "education"/"skills"
    # appear in many non-resume docs (e.g. an education textbook filename).
    "resume":    ("resume", "curriculum vitae", "work experience", "employment history",
                  "references available upon"),
    "textbook":  ("textbook", "chapter", "exercises", "learning objectives", " book "),
    "code_manual": ("readme", "installation", "api reference", "```", "def ", "import "),
}


def _norm(t: Optional[str]) -> str:
    t = (t or "").strip().lower().replace(" ", "_")
    aliases = {"code": "code_manual", "tech_manual": "code_manual", "cv": "resume"}
    t = aliases.get(t, t)
    return t if t in CHUNKER_PROFILES else "general"


def _block_text(b: Dict[str, Any]) -> str:
    return str(b.get("text") or b.get("md") or b.get("summary") or b.get("label") or "")


def classify_doc_type(
    blocks: List[Dict[str, Any]],
    filename: str = "",
    forced: Optional[str] = None,
) -> str:
    """Return a doc_type key. A user-supplied `forced` type (not 'Auto') wins."""
    # 1) explicit user override
    if forced and forced.strip().lower() not in ("auto (detect)", "auto", "", "general"):
        return _norm(forced)

    name = (filename or "").lower()
    sample = " ".join(_block_text(b) for b in blocks[:12]).lower()

    def hit(kws):
        return any(k in name for k in kws) or any(k in sample for k in kws)

    # 2) strong keyword signal (filename or opening content)
    for dtype in ("financial", "legal", "resume", "medical", "academic", "textbook", "code_manual"):
        if hit(_KEYWORDS[dtype]):
            # table density disambiguates financial vs a doc that merely says "revenue"
            if dtype == "financial" and _table_density(blocks) < 0.05 and "invoice" not in name:
                continue
            return dtype

    # 3) structural signal: heavily tabular docs read like financial/data reports.
    # (No "short doc → resume" guess — that mislabels short generic docs; resume
    # relies on its keywords above.)
    if _table_density(blocks) > 0.15:
        return "financial"

    return "general"


def _table_density(blocks: List[Dict[str, Any]]) -> float:
    if not blocks:
        return 0.0
    tables = sum(1 for b in blocks if b.get("type") == "table")
    return tables / len(blocks)


def profile_for(doc_type: str) -> Dict[str, Any]:
    """Chunker kwargs for a doc_type (falls back to the general profile)."""
    return dict(CHUNKER_PROFILES.get(_norm(doc_type), CHUNKER_PROFILES["general"]))
