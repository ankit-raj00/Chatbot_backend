"""
Structure-aware chunking over the parser's typed blocks.

The PDF parser emits typed blocks instead of a flat markdown string:

    {"type": "heading",   "level": 2, "text": ..., "page": 7, "bbox": [...]}
    {"type": "paragraph", "text": ...}
    {"type": "list",      "text": ...}
    {"type": "table",     "md": ...}
    {"type": "figure",    "filename": ..., "summary": ...}
    {"type": "callout",   "label": "Activity Research", "blocks": [...]}
    {"type": "caption" | "formula" | "header_footer", ...}

Chunking rules (why each matters for retrieval):
  * A heading STARTS a new chunk — a chunk never straddles a section boundary.
  * `heading_path` breadcrumbs ride along as metadata, so a chunk retrieved on
    its own still knows "Chapter 1 > Socialisation > Status and role".
  * callout / table / figure are ATOMIC — an Activity box or a table is a single
    semantic unit; splitting it mid-way destroys the answer it contains.
  * Prose accumulates up to `max_chars`, then flushes on a paragraph boundary
    (never mid-sentence), with optional overlap for context continuity.
  * `header_footer` blocks (running heads, page numbers) are dropped — they are
    noise that pollutes embeddings.
"""
from typing import Any, Dict, Iterable, List, Optional

ATOMIC = {"callout", "table", "figure"}
SKIP = {"header_footer"}


def _block_text(b: Dict[str, Any]) -> str:
    """Render one block to the text that will be embedded."""
    t = b.get("type")
    if t == "heading":
        return f"{'#' * int(b.get('level') or 2)} {b.get('text', '')}".strip()
    if t == "table":
        return (b.get("md") or "").strip()
    if t == "formula":
        return (b.get("latex") or "").strip()
    if t == "figure":
        # the summary is what makes a figure retrievable at all
        name = b.get("filename") or ""
        return f"[Figure {name}] {b.get('summary', '')}".strip()
    if t == "callout":
        label = b.get("label") or "Box"
        inner = "\n".join(_block_text(s) for s in (b.get("blocks") or []))
        return f"{label}\n{inner}".strip()
    return (b.get("text") or "").strip()


def _bbox_union(boxes: List[Optional[list]]) -> Optional[list]:
    bb = [b for b in boxes if b]
    if not bb:
        return None
    return [min(b[0] for b in bb), min(b[1] for b in bb),
            max(b[2] for b in bb), max(b[3] for b in bb)]


def chunk_blocks(
    blocks: Iterable[Dict[str, Any]],
    max_chars: int = 1200,
    min_chars: int = 200,
    overlap_chars: int = 0,
    split_on_headings: bool = True,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Turn typed blocks into retrieval chunks.

    Returns [{"text": str, "metadata": {...}}] where metadata carries
    heading_path, page range, bbox and the block types the chunk covers.

    split_on_headings=False keeps the whole document flowing (bounded only by
    max_chars) — used for doc-types that are one semantic unit, e.g. resumes.
    """
    base_metadata = dict(base_metadata or {})
    chunks: List[Dict[str, Any]] = []
    heading_path: List[str] = []          # index 0 == H1

    buf: List[str] = []
    buf_blocks: List[Dict[str, Any]] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n\n".join(x for x in buf if x).strip()
        if not text:
            buf.clear(); buf_blocks.clear()
            return
        pages = [b.get("page") for b in buf_blocks if b.get("page") is not None]
        # figures whose crop lives in this chunk — keep filename + Cloudinary URL
        figures = [
            {"filename": b.get("filename"), "url": b.get("url")}
            for b in buf_blocks if b.get("type") == "figure" and b.get("filename")
        ]
        meta = {
            **base_metadata,
            "heading_path": list(heading_path),
            "section": heading_path[-1] if heading_path else None,
            "page": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "block_types": sorted({b.get("type") for b in buf_blocks if b.get("type")}),
            "bbox": _bbox_union([b.get("bbox") for b in buf_blocks]),
        }
        if figures:
            meta["figures"] = figures
        chunks.append({"text": text, "metadata": meta})
        # carry a tail of context into the next chunk when asked
        if overlap_chars > 0 and len(text) > overlap_chars:
            tail = text[-overlap_chars:]
            buf.clear(); buf_blocks.clear()
            buf.append(tail)
        else:
            buf.clear(); buf_blocks.clear()

    def cur_len() -> int:
        return sum(len(x) + 2 for x in buf)

    for b in blocks:
        btype = b.get("type")
        if btype in SKIP:
            continue

        text = _block_text(b)
        if not text:
            continue

        if btype == "heading":
            # a heading begins a new chunk (unless whole-doc mode)
            if split_on_headings:
                flush()
            lvl = max(1, min(6, int(b.get("level") or 2)))
            heading_path[:] = heading_path[: lvl - 1]
            while len(heading_path) < lvl - 1:
                heading_path.append("")
            heading_path.append(b.get("text", "").strip())
            buf.append(text)
            buf_blocks.append(b)
            if not split_on_headings and cur_len() >= max_chars:
                flush()
            continue

        if btype in ATOMIC:
            # never split these; give them their own chunk when the buffer is
            # already substantial, otherwise let them ride with nearby prose.
            if cur_len() >= min_chars or len(text) >= max_chars:
                flush()
                # re-emit the heading context so the atomic chunk isn't orphaned
                if heading_path and heading_path[-1]:
                    buf.append(f"{'#' * len(heading_path)} {heading_path[-1]}")
            buf.append(text)
            buf_blocks.append(b)
            if cur_len() >= max_chars:
                flush()
            continue

        # ordinary prose
        if cur_len() + len(text) > max_chars and cur_len() >= min_chars:
            flush()
            if heading_path and heading_path[-1]:
                buf.append(f"{'#' * len(heading_path)} {heading_path[-1]}")
        buf.append(text)
        buf_blocks.append(b)

    flush()
    return chunks
