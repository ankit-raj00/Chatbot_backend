"""
Hybrid scanned-PDF → Markdown parser for RAG.

Pipeline (per page, when scanned/complex):
    PyMuPDF render → DocLayout-YOLO regions → crop → OmniRoute/Gemini per region → merge

Modes (so the test UI can compare cost/time on the same PDF):
    - digital    : PyMuPDF text extraction only (no model, ~free)
    - wholepage  : render page → single Gemini VLM call
    - region     : DocLayout-YOLO → crop → Gemini per region (the hybrid)
    - auto       : per-page — digital if it has real text, else region

Returns markdown + extracted figure images (base64) + timing + token/cost.
"""
import os
import io
import json
import time
import base64
import threading
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OMNI = OpenAI(
    base_url=os.getenv("OMNIROUTE_BASE_URL", "http://host.docker.internal:20128/v1"),
    api_key=os.getenv("OMNIROUTE_API_KEY", ""),
    max_retries=5,     # a dropped call = permanently lost page text, so retry hard
    timeout=120,
)
VISION_MODEL = os.getenv("VISION_MODEL", "antigravity/gemini-3.5-flash-medium")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "antigravity/claude-sonnet-5")
JUDGE_USD_IN = float(os.getenv("JUDGE_USD_PER_1M_INPUT", "3.0"))
JUDGE_USD_OUT = float(os.getenv("JUDGE_USD_PER_1M_OUTPUT", "15.0"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "/models/doclayout.pt")
RENDER_DPI = int(os.getenv("RENDER_DPI", "180"))
# base64 crops are handy for the eval UI but bloat ingestion responses.
# Per-request `include_b64` overrides this default; threadlocal carries it in.
INCLUDE_FIGURE_B64 = os.getenv("INCLUDE_FIGURE_B64", "true").lower() == "true"
_tls = threading.local()


def _want_b64() -> bool:
    return getattr(_tls, "include_b64", INCLUDE_FIGURE_B64)
# Region VLM calls run SEQUENTIALLY by default: parallel calls tripped the
# gateway's rate limits, and failed calls silently dropped page content.
# Set REGION_WORKERS > 1 only if the gateway can take the concurrency.
REGION_WORKERS = int(os.getenv("REGION_WORKERS", "1"))
_cost_lock = threading.Lock()
# Page-level structure pass: one whole-page call that fixes reading order and
# assigns block roles/heading levels (region crops alone cannot know either).
STRUCTURE_PASS = os.getenv("STRUCTURE_PASS", "true").lower() == "true"
STRUCTURE_MODEL = os.getenv("STRUCTURE_MODEL", os.getenv("VISION_MODEL", "antigravity/gemini-3.5-flash-medium"))
STRUCTURE_DEBUG = os.getenv("STRUCTURE_DEBUG", "").lower() in ("1", "true")

# ── Cloudinary: figures are uploaded so callers get a durable URL instead of
# a fat base64 payload (base64 is still returned for the eval UI unless off).
_cloudinary_ready = False
if os.getenv("CLOUDINARY_CLOUD_NAME"):
    try:
        import cloudinary
        import cloudinary.uploader
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True,
        )
        _cloudinary_ready = True
    except Exception as _e:  # pragma: no cover
        print(f"[cloudinary] disabled: {_e}", flush=True)


def _upload_figure(png_bytes: bytes, public_id: str) -> Optional[str]:
    """Upload a figure crop and return its Cloudinary URL (None if unavailable)."""
    if not _cloudinary_ready:
        return None
    try:
        res = cloudinary.uploader.upload(
            png_bytes, public_id=public_id, folder="pdf-parser/figures",
            resource_type="image", overwrite=True,
        )
        return res.get("secure_url") or res.get("url")
    except Exception as e:
        print(f"[cloudinary] upload failed for {public_id}: {e}", flush=True)
        return None


def _upload_page_thumb(png_bytes: bytes, public_id: str) -> Optional[str]:
    """Upload a rendered page thumbnail for the parser dashboard's compare
    view — a durable CDN URL instead of a fat base64 blob in Mongo."""
    if not _cloudinary_ready:
        return None
    try:
        res = cloudinary.uploader.upload(
            png_bytes, public_id=public_id, folder="pdf-parser/page-thumbs",
            resource_type="image", overwrite=True,
        )
        return res.get("secure_url") or res.get("url")
    except Exception as e:
        print(f"[cloudinary] page thumb upload failed for {public_id}: {e}", flush=True)
        return None
# A 47-block page needs far more than 2000 tokens of ordering JSON.
STRUCTURE_MAX_TOKENS = int(os.getenv("STRUCTURE_MAX_TOKENS", "8000"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1024"))
# rough Gemini-Flash-class rates (USD / 1M tokens) — override via env for accuracy
USD_IN = float(os.getenv("USD_PER_1M_INPUT", "0.10"))
USD_OUT = float(os.getenv("USD_PER_1M_OUTPUT", "0.40"))

# DocLayout-YOLO (DocStructBench) class names → how we treat them
KIND_PROMPTS = {
    # NOTE: heading '#' markers are added by us (level comes from relative size),
    # so the model must return the bare heading text only.
    "title":            "Transcribe this heading's text exactly. Output ONLY the text itself — no markdown symbols, no '#', no quotes.",
    "plain text":       ("Transcribe ALL text in this image exactly, as GitHub-flavored markdown. "
                         "Preserve paragraph breaks. Render bullet points as '- ' items and numbered "
                         "lists as '1. ' items. Use **bold** for terms that are bold/emphasised in the "
                         "image. If the block is a table, render a markdown table. "
                         "Output only the transcription — no commentary."),
    "table":            "Transcribe this table as a GitHub-flavored markdown table. Preserve every cell, number and header exactly. Output only the table.",
    "table_caption":    "Transcribe this caption text exactly. Output only the text.",
    "table_footnote":   "Transcribe this footnote text exactly. Output only the text.",
    "isolate_formula":  "Transcribe this mathematical formula as LaTeX. Output ONLY the LaTeX wrapped in $$ ... $$.",
    "formula_caption":  "Transcribe this text exactly. Output only the text.",
    "figure_caption":   "Transcribe this figure caption exactly. Output only the text.",
}
# regions we drop (page numbers, headers/footers, running marks)
DROP = {"abandon"}
# regions saved as images (not sent for full OCR)
FIGURE = {"figure"}

_model = None
_region_ok = None  # None=unknown, True/False once probed


def region_available() -> bool:
    """True if DocLayout-YOLO (torch) is installed on this box."""
    global _region_ok
    if _region_ok is None:
        try:
            import doclayout_yolo  # noqa: F401
            _region_ok = True
        except Exception:
            _region_ok = False
    return _region_ok


def _get_model():
    """Lazy-load YOLO so the digital/wholepage modes work even if torch is heavy."""
    global _model
    if _model is None:
        from doclayout_yolo import YOLOv10
        _model = YOLOv10(YOLO_WEIGHTS)
    return _model


# ── Helpers ─────────────────────────────────────────────────────────────────
def _page_image(page, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render a PDF page to a BGR numpy image."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:      # RGBA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:    # RGB → BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:               # gray → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _vlm(img_bgr: np.ndarray, prompt: str, cost: Dict[str, int]) -> str:
    """One OmniRoute/Gemini vision call on an image crop. Accumulates token cost."""
    b64 = _png_b64(img_bgr)
    resp = OMNI.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    u = getattr(resp, "usage", None)
    with _cost_lock:                    # called from worker threads
        if u:
            cost["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            cost["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
        cost["vlm_calls"] += 1
    return (resp.choices[0].message.content or "").strip()


def _overlap(a: list, b: list):
    """Return (IoU, containment) for two xyxy boxes. containment = inter / smaller-area."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aarea = max(1, (ax2 - ax1) * (ay2 - ay1))
    barea = max(1, (bx2 - bx1) * (by2 - by1))
    iou = inter / (aarea + barea - inter)
    contain = inter / min(aarea, barea)
    return iou, contain


def _area(b: list) -> float:
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def _dedup_regions(regions: List[dict]) -> List[dict]:
    """
    Drop duplicate detections (keep highest-confidence).

    A duplicate is either near-identical (high IoU) or mostly nested inside a
    kept box (e.g. YOLO emitting 'title' and 'plain text' over the same block,
    or a box whose text the enclosing region already transcribes).
    """
    kept: List[dict] = []
    for r in sorted(regions, key=lambda x: x["conf"], reverse=True):
        drop = False
        for k in kept:
            iou, contain = _overlap(r["bbox"], k["bbox"])
            if iou > 0.5 or contain > 0.7:
                drop = True
                break
        if not drop:
            kept.append(r)
    return kept


def _reading_order(regions: List[dict]) -> List[dict]:
    """
    Column-aware order — but ONLY when the layout really is multi-column.
    A true column layout has >=2 x-clusters that each hold >=2 regions AND sit
    side-by-side (their y-ranges overlap). Otherwise (covers, posters, stacked
    layouts) fall back to plain top->bottom, which is the correct reading order.
    """
    if len(regions) <= 1:
        return list(regions)
    span = (max(r["bbox"][2] for r in regions) - min(r["bbox"][0] for r in regions))
    gap = max(60, span * 0.14)
    def xc(r):
        return (r["bbox"][0] + r["bbox"][2]) / 2

    # NOTE: sort by key only — never put dicts in the tuple, they aren't comparable
    # when two regions share the same x-centre (common on aligned columns).
    items = sorted(regions, key=xc)
    cols = [[items[0]]]
    for r in items[1:]:
        if xc(r) - xc(cols[-1][-1]) > gap:   # big horizontal gap → candidate new column
            cols.append([r])
        else:
            cols[-1].append(r)

    def yrange(col):
        return min(r["bbox"][1] for r in col), max(r["bbox"][3] for r in col)

    multi = len(cols) >= 2 and all(len(c) >= 2 for c in cols)
    if multi:  # adjacent columns must sit side-by-side AND have comparable heights
        for a, b in zip(cols, cols[1:]):
            a0, a1 = yrange(a)
            b0, b1 = yrange(b)
            inter = max(0, min(a1, b1) - max(a0, b0))
            ha, hb = (a1 - a0) or 1, (b1 - b0) or 1
            shorter = min(ha, hb)
            # Real text columns run roughly parallel down the page. A short block
            # nested inside a tall one (covers/posters) is NOT a column.
            if inter / shorter < 0.5 or shorter / max(ha, hb) < 0.5:
                multi = False
                break

    if not multi:
        return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    ordered: List[dict] = []
    for col in cols:                      # columns already left→right
        ordered.extend(sorted(col, key=lambda r: r["bbox"][1]))
    return ordered


def _is_blank(crop: np.ndarray, page_area: float = 0.0) -> bool:
    """True if the crop is effectively empty (blank / scratched scanner page)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg = float(np.median(g))
    ink = float((np.abs(g.astype(np.int16) - bg) > 40).mean())
    if ink < 0.02:
        return True
    # A near-empty region covering most of the page is a blank scan page whose
    # scratches/speckle push ink above the flat threshold — not a real figure.
    if page_area and ink < 0.06 and (crop.shape[0] * crop.shape[1]) > 0.5 * page_area:
        return True
    return False


# ── Per-page strategies ───────────────────────────────────────────────────────
def _page_digital(page) -> Optional[str]:
    """Return embedded text as markdown, or None if the page has too little text."""
    txt = page.get_text("text").strip()
    if len(txt) < 30:
        return None
    return txt


def _page_wholepage(page, cost: Dict[str, int]) -> str:
    img = _page_image(page)
    return _vlm(img, "Transcribe this document page to clean markdown. Preserve headings, "
                     "paragraphs, lists and tables (as GFM tables). Describe any figure in one "
                     "sentence. Output only the markdown.", cost)


import re as _re

# The VLM tells us when a crop has no real content — use that to drop scan
# artefacts (table surface behind the book, blank pages, speckle) that the
# pixel-level blank test can't catch because they are textured.
_NOISE_CAPTION = _re.compile(
    r"(illegible|blank|empty|featureless|no (?:visible|discernible|legible|readable)"
    r"|scanner (?:noise|artifact)|scratched (?:grey|gray|white) surface)", _re.I)


def _is_noise_caption(c: str) -> bool:
    return bool(_NOISE_CAPTION.search(c or ""))


FIGURE_PROMPT = (
    "Describe ONLY what is clearly and unambiguously visible in this image crop. "
    "If it is blank, faint, noisy, or the text is illegible, say exactly that in a few words "
    "(e.g. 'faint illegible ink stamp'). NEVER guess or invent any text, names, dates, numbers, "
    "logos, or institutions that you cannot clearly read. One short factual sentence."
)


def _assign_heading_levels(regions: List[dict]) -> None:
    """Heading level from relative size: biggest title on the page = H1, next = H2, rest = H3."""
    titles = [r for r in regions if r["kind"] == "title"]
    if not titles:
        return
    heights = sorted({r["bbox"][3] - r["bbox"][1] for r in titles}, reverse=True)
    for r in titles:
        i = heights.index(r["bbox"][3] - r["bbox"][1])
        r["_level"] = 1 if i == 0 else (2 if i == 1 else 3)


_STRUCTURE_PROMPT = (
    "You are given a scanned page image and the list of text blocks that were extracted from it "
    "(each with an id, its bounding box [x1,y1,x2,y2], its detected kind and a short snippet).\n"
    "Using the PAGE IMAGE to understand the real layout (columns, boxes, sidebars), return:\n"
    "  1. the correct READING ORDER of the blocks — the order a human reads them,\n"
    "  2. each block's role, and heading level where relevant.\n"
    "roles: heading | paragraph | list | table | figure | caption | header_footer\n"
    "For 'heading' give level 1-3 (1 = most prominent on the page); otherwise level null.\n"
    "CALLOUT PANELS: if several blocks are visually enclosed in the SAME shaded/boxed panel "
    "(e.g. an 'Activity' box, a numbered 'Box 7' panel, a tinted sidebar), give every block in "
    "that panel the SAME integer \"group\" id, and put the panel's label (e.g. 'Activity Research', "
    "'Box 7 Postmodern society') in \"label\" on the FIRST block of that group. Blocks that are "
    "NOT inside a panel must have \"group\": null.\n"
    'Return ONLY compact JSON: {"order":[{"id":<int>,"role":"<role>","level":<int|null>,'
    '"group":<int|null>,"label":"<string|null>"}, ...]}\n'
    "Include EVERY id exactly once. Do not invent ids.\n\nBLOCKS:\n"
)


def _structure_pass(img: np.ndarray, jobs: List[dict], texts: List[str],
                    cost: Dict[str, int]) -> Optional[List[dict]]:
    """
    One whole-page call that returns reading order + roles/levels for the blocks.
    Returns a validated list of {id, role, level}, or None to keep geometric order.
    Never loses blocks: any id the model omits is appended in geometric order.
    """
    if not jobs:
        return None
    manifest = [{
        "id": i,
        "kind": j["kind"],
        "bbox": j["region"]["bbox"],
        "snippet": " ".join((t or "").split())[:90],
    } for i, (j, t) in enumerate(zip(jobs, texts))]

    try:
        b64 = _png_b64(_downscale(img, 1400))
        resp = OMNI.chat.completions.create(
            model=STRUCTURE_MODEL, max_tokens=STRUCTURE_MAX_TOKENS, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _STRUCTURE_PROMPT + json.dumps(manifest)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        )
        u = getattr(resp, "usage", None)
        with _cost_lock:
            if u:
                cost["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                cost["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
            cost["structure_calls"] = cost.get("structure_calls", 0) + 1
        txt = (resp.choices[0].message.content or "").strip()
        m = _re.search(r"\{.*\}", txt, _re.DOTALL)
        if not m:
            if STRUCTURE_DEBUG:
                print(f"[structure] no JSON in reply ({len(txt)} chars): {txt[:160]!r}", flush=True)
            return None
        raw = json.loads(m.group(0)).get("order") or []
    except Exception as e:
        if STRUCTURE_DEBUG:
            print(f"[structure] FAILED blocks={len(jobs)}: {type(e).__name__}: {e}", flush=True)
        return None

    out, used = [], set()
    for o in raw:
        try:
            i = int(o["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < len(jobs) and i not in used:
            used.add(i)
            lvl = o.get("level")
            grp = o.get("group")
            lbl = o.get("label")
            out.append({"id": i, "role": str(o.get("role") or ""),
                        "level": int(lvl) if isinstance(lvl, (int, float)) else None,
                        "group": int(grp) if isinstance(grp, (int, float)) else None,
                        "label": str(lbl).strip() if lbl else None})
    if not out:
        if STRUCTURE_DEBUG:
            print(f"[structure] no valid ids from {len(raw)} entries", flush=True)
        return None
    # A dense page the model only partially ordered is worse than plain geometry:
    # trust it only when it accounted for most blocks.
    if len(out) < 0.8 * len(jobs):
        if STRUCTURE_DEBUG:
            print(f"[structure] REJECT: named {len(out)}/{len(jobs)} blocks → keep geometric", flush=True)
        return None
    # never drop content the model forgot to mention — keep it near its neighbours
    missing = [i for i in range(len(jobs)) if i not in used]
    for i in missing:
        pos = next((k for k, o in enumerate(out) if o["id"] > i), len(out))
        out.insert(pos, {"id": i, "role": "", "level": None, "group": None, "label": None})
    if STRUCTURE_DEBUG:
        print(f"[structure] ok: {len(used)}/{len(jobs)} named, {len(missing)} reinserted", flush=True)
    return out


def _regions_markdown(img: np.ndarray, page_no: int, cost: Dict[str, int],
                      figs: List[dict], seen: set):
    """
    YOLO → dedup → reading-order → per-region VLM on ONE image (page or half-page).

    Returns (markdown, blocks). `blocks` is the typed structure used for
    structure-aware chunking: heading/paragraph/table/figure/caption/formula,
    each with page + bbox, emitted in true reading order.
    Region VLM calls run in parallel but results are re-assembled IN ORDER.
    """
    h, w = img.shape[:2]
    model = _get_model()
    det = model.predict(img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, device="cpu", verbose=False)[0]
    names = det.names

    regions = []
    for b in det.boxes:
        cls = names[int(b.cls)]
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append({"kind": cls, "bbox": [x1, y1, x2, y2], "conf": float(b.conf)})

    regions = _dedup_regions(regions)      # kill overlapping/nested duplicates
    regions = _reading_order(regions)      # column-aware order (figures included, in place)
    _assign_heading_levels(regions)

    page_area = float(h * w)
    jobs: List[dict] = []
    for r in regions:
        kind = r["kind"]
        x1, y1, x2, y2 = r["bbox"]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if kind in FIGURE:
            if _is_blank(crop, page_area):     # blank/scratched scanner page → not a figure
                continue
            jobs.append({"kind": "figure", "crop": crop, "prompt": FIGURE_PROMPT, "region": r})
            continue
        if kind in DROP:
            # 'abandon' holds real content: running headers, page numbers,
            # publisher lines, website boxes. Keep them; skip only specks.
            if (x2 - x1) * (y2 - y1) < 0.0008 * page_area:
                continue
            prompt = KIND_PROMPTS["plain text"]
        else:
            prompt = KIND_PROMPTS.get(kind)
        if not prompt:
            continue
        jobs.append({"kind": kind, "crop": crop, "prompt": prompt, "region": r})

    def _work(j):
        try:
            return _vlm(j["crop"], j["prompt"], cost)
        except Exception as e:
            return f"_[region {j['kind']} failed: {e}]_"

    if not jobs:
        results = []
    elif REGION_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=REGION_WORKERS) as ex:
            results = list(ex.map(_work, jobs))   # ex.map preserves input order
    else:
        results = [_work(j) for j in jobs]        # sequential — no rate-limit drops

    # Page-level structure pass: fixes cross-column reading order and assigns
    # roles/heading levels, which an isolated crop can never determine.
    order = _structure_pass(img, jobs, results, cost) if STRUCTURE_PASS else None
    if order:
        seq = [(jobs[o["id"]], results[o["id"]], o) for o in order]
    else:
        seq = [(j, t, None) for j, t in zip(jobs, results)]

    # rendered items: (group_id, label, markdown, block) — grouped into callouts below
    rendered: List[tuple] = []
    for j, text, meta in seq:
        r = j["region"]
        text = (text or "").strip()
        grp = (meta or {}).get("group")
        lbl = (meta or {}).get("label")

        if j["kind"] == "figure":
            if not text or _is_noise_caption(text):   # nothing readable → scan artefact
                continue
            fname = f"page{page_no:02d}_fig{len(figs) + 1}.png"
            ok, buf = cv2.imencode(".png", j["crop"])
            png_bytes = buf.tobytes() if ok else b""
            url = _upload_figure(png_bytes, f"{_uuid.uuid4().hex[:12]}_{fname[:-4]}") if png_bytes else None

            fig = {"id": fname[:-4], "filename": fname, "page": page_no,
                   "kind": "figure", "caption": text, "url": url}
            if _want_b64():
                fig["b64"] = base64.b64encode(png_bytes).decode() if png_bytes else ""
            figs.append(fig)

            # Figure sits at its true position in the page flow, with filename + summary.
            link = url or fname
            rendered.append((grp, lbl,
                             f"**Figure: `{fname}`**\n\n*Summary:* {text}\n\n![{fname}]({link})",
                             {"type": "figure", "filename": fname, "url": url,
                              "summary": text, "page": page_no, "bbox": r["bbox"]}))
            continue

        if not text:
            continue
        norm = " ".join(text.lower().split())[:200]
        if len(norm) > 25 and norm in seen:   # drop duplicated text blocks
            continue
        seen.add(norm)

        kind = j["kind"]
        role = (meta or {}).get("role") or ""
        # The structure pass saw the whole page, so its role wins over the
        # per-crop class; fall back to the detected kind when it said nothing.
        is_heading = role == "heading" or (not role and kind == "title")
        if is_heading and grp is None:
            lvl = (meta or {}).get("level") or r.get("_level", 2)
            lvl = min(6, max(1, int(lvl)))
            clean = " ".join(text.split())
            rendered.append((grp, lbl, "#" * lvl + " " + clean,
                             {"type": "heading", "level": lvl, "text": clean,
                              "page": page_no, "bbox": r["bbox"]}))
        elif is_heading:
            # Inside a callout panel a "heading" is just the panel's title — emitting
            # it as '###' invented phantom sections (e.g. 'Status and role' twice).
            clean = " ".join(text.split())
            rendered.append((grp, lbl or clean, f"**{clean}**",
                             {"type": "text", "text": clean, "page": page_no, "bbox": r["bbox"]}))
        elif role == "header_footer":
            rendered.append((grp, lbl, f"*{text}*",
                             {"type": "header_footer", "text": text,
                              "page": page_no, "bbox": r["bbox"]}))
        elif role == "list":
            rendered.append((grp, lbl, text,
                             {"type": "list", "text": text, "page": page_no, "bbox": r["bbox"]}))
        elif kind == "table" or role == "table":
            rendered.append((grp, lbl, text,
                             {"type": "table", "md": text, "page": page_no, "bbox": r["bbox"]}))
        elif kind == "isolate_formula":
            rendered.append((grp, lbl, text,
                             {"type": "formula", "latex": text, "page": page_no, "bbox": r["bbox"]}))
        elif "caption" in kind:
            rendered.append((grp, lbl, f"*{text}*",
                             {"type": "caption", "text": text, "page": page_no, "bbox": r["bbox"]}))
        else:
            rendered.append((grp, lbl, text,
                             {"type": "paragraph", "text": text, "page": page_no, "bbox": r["bbox"]}))

    # ── Merge consecutive same-group items into ONE atomic callout block ──────
    parts: List[str] = []
    blocks: List[dict] = []
    i = 0
    while i < len(rendered):
        grp, lbl, md, blk = rendered[i]
        if grp is None:
            parts.append(md)
            blocks.append(blk)
            i += 1
            continue
        members, label = [], lbl
        while i < len(rendered) and rendered[i][0] == grp:
            if rendered[i][1] and not label:
                label = rendered[i][1]
            members.append(rendered[i])
            i += 1
        # The panel's badge/title blocks ('Activity', 'Status and role') are already
        # carried by `label` — drop them from the body so it isn't stated twice.
        def _norm(s: str) -> str:
            return " ".join((s or "").replace("*", "").replace("#", "").lower().split())

        nlabel = _norm(label)
        keep = [m for m in members
                if not (nlabel and _norm(m[2]) and _norm(m[2]) in nlabel)]
        if not keep:                    # panel was nothing but its title
            keep = members
        inner = "\n\n".join(m[2] for m in keep if m[2])
        sub = [m[3] for m in keep]
        body = (f"**{label}**\n\n{inner}" if label else inner)
        # blockquote keeps the panel visually distinct and atomic in markdown
        parts.append("\n".join(("> " + ln) if ln.strip() else ">" for ln in body.split("\n")))
        bbs = [b["bbox"] for b in sub if b.get("bbox")]
        bbox = ([min(b[0] for b in bbs), min(b[1] for b in bbs),
                 max(b[2] for b in bbs), max(b[3] for b in bbs)] if bbs else None)
        blocks.append({"type": "callout", "label": label or "", "page": page_no,
                       "bbox": bbox, "blocks": sub})

    return "\n\n".join(parts), blocks


def _page_region(page, page_no: int, cost: Dict[str, int], images: List[dict]):
    """Hybrid parse. Book spreads (wide pages) are split into left/right physical pages."""
    img = _page_image(page)
    h, w = img.shape[:2]
    seen: set = set()
    if w > h * 1.25:   # two-page spread → process left page fully, then right page
        mid = w // 2
        l_md, l_blocks = _regions_markdown(img[:, :mid], page_no, cost, images, seen)
        r_md, r_blocks = _regions_markdown(img[:, mid:], page_no, cost, images, seen)
        return (l_md + "\n\n" + r_md).strip(), l_blocks + r_blocks
    return _regions_markdown(img, page_no, cost, images, seen)


def _downscale(img_bgr: np.ndarray, max_edge: int = 1100) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return img_bgr
    s = max_edge / m
    return cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _page_thumb(page) -> str:
    """A downscaled PNG of the rendered page, for the side-by-side preview."""
    return _png_b64(_downscale(_page_image(page, dpi=110), 1100))


def _judge(page_img_bgr: np.ndarray, markdown: str, cost: Dict[str, int]) -> Dict[str, Any]:
    """LLM-as-judge: score how faithfully `markdown` represents the page image."""
    import re
    import json as _json
    b64 = _png_b64(_downscale(page_img_bgr, 1400))
    prompt = (
        "You are grading a document parser. The MARKDOWN below was extracted by the parser "
        "from the attached page image. Compare them and grade how faithfully the markdown "
        "represents the page.\n"
        "Score 0-100 (100 = all text, tables, figures and reading order captured, no hallucinations). "
        "List concrete issues (missed text, broken/incorrect tables, wrong reading order, missed or "
        "hallucinated figures, duplicated content).\n"
        'Return ONLY compact JSON: {"score": <int>, "verdict": "<one line>", "issues": ["...","..."]}.\n\n'
        # Must be large enough to hold a full 2-page spread, otherwise the judge
        # reports the tail of the page as "missing" when it simply never saw it.
        f"EXTRACTED MARKDOWN:\n{markdown[:24000]}"
    )
    try:
        r = OMNI.chat.completions.create(
            model=JUDGE_MODEL, max_tokens=500, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        )
        u = getattr(r, "usage", None)
        if u:
            cost["judge_input_tokens"] = cost.get("judge_input_tokens", 0) + (getattr(u, "prompt_tokens", 0) or 0)
            cost["judge_output_tokens"] = cost.get("judge_output_tokens", 0) + (getattr(u, "completion_tokens", 0) or 0)
        cost["judge_calls"] = cost.get("judge_calls", 0) + 1
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            data = _json.loads(m.group(0))
            data["score"] = int(data["score"]) if data.get("score") is not None else None
            return data
        return {"score": None, "verdict": txt[:160], "issues": []}
    except Exception as e:
        return {"score": None, "verdict": f"judge failed: {e}", "issues": []}


# ── Entry points ──────────────────────────────────────────────────────────────
def parse_pdf_iter(path: str, mode: str = "auto", max_pages: int = 0, judge: bool = False):
    """
    Generator: yields one event per page as it completes, then a final 'done'.
      {"type":"start", pages, mode}
      {"type":"page",  page, mode, seconds, markdown, page_image(b64), figures[], judge?, cost}
      {"type":"done",  pages, timing, cost, judge_avg}
    """
    t0 = time.time()
    doc = fitz.open(path)
    n = doc.page_count
    if max_pages and max_pages > 0:
        n = min(n, max_pages)

    cost = {"input_tokens": 0, "output_tokens": 0, "vlm_calls": 0}
    per_page = []
    scores = []
    yield {"type": "start", "pages": n, "mode": mode}

    for i in range(n):
        page = doc[i]
        pt0 = time.time()
        used = mode
        if mode == "auto":
            used = "digital" if _page_digital(page) is not None else "region"

        page_figs: List[dict] = []
        page_blocks: List[dict] = []
        try:
            if used == "digital":
                md = _page_digital(page) or _page_wholepage(page, cost)
            elif used == "wholepage":
                md = _page_wholepage(page, cost)
            elif used == "region":
                md, page_blocks = _page_region(page, i + 1, cost, page_figs)
            else:
                md = _page_digital(page) or ""
        except Exception as e:
            md = f"_[page {i+1} failed: {e}]_"

        secs = round(time.time() - pt0, 2)
        per_page.append({"page": i + 1, "mode": used, "seconds": secs})
        try:
            thumb = _page_thumb(page)
        except Exception:
            thumb = None
        thumb_url = None
        if thumb:
            try:
                thumb_url = _upload_page_thumb(base64.b64decode(thumb), f"{_uuid.uuid4().hex[:12]}_p{i+1}")
            except Exception:
                thumb_url = None

        judge_res = None
        if judge:
            try:
                judge_res = _judge(_page_image(page, dpi=150), md, cost)
                if judge_res.get("score") is not None:
                    scores.append(judge_res["score"])
            except Exception as e:
                judge_res = {"score": None, "verdict": f"judge error: {e}", "issues": []}

        yield {
            "type": "page", "page": i + 1, "mode": used, "seconds": secs,
            "markdown": md, "page_image": thumb, "page_image_url": thumb_url,
            "figures": page_figs,
            "blocks": page_blocks,          # typed structure for chunking
            "judge": judge_res, "cost": dict(cost),
        }

    doc.close()
    total = round(time.time() - t0, 2)
    est_usd = round(cost["input_tokens"] / 1e6 * USD_IN + cost["output_tokens"] / 1e6 * USD_OUT, 6)
    judge_est = round(cost.get("judge_input_tokens", 0) / 1e6 * JUDGE_USD_IN
                      + cost.get("judge_output_tokens", 0) / 1e6 * JUDGE_USD_OUT, 6)
    yield {
        "type": "done", "pages": n,
        "timing": {"total_s": total, "per_page": per_page},
        "judge_avg": round(sum(scores) / len(scores), 1) if scores else None,
        "cost": {**cost, "est_usd": est_usd, "judge_est_usd": judge_est,
                 "rates": {"usd_per_1m_input": USD_IN, "usd_per_1m_output": USD_OUT}},
    }


def parse_pdf(path: str, mode: str = "auto", max_pages: int = 0, judge: bool = False,
              include_b64: bool = True) -> Dict[str, Any]:
    """Non-streaming: aggregate the per-page generator into one result."""
    _tls.include_b64 = include_b64
    page_md: List[str] = []
    images: List[dict] = []
    blocks: List[dict] = []
    # Per-page preview (rendered page image + exactly what was extracted from
    # it) — the streaming path already computes page_image per page for its
    # live SSE preview; this just also keeps it instead of discarding it, so
    # callers (the parser dashboard) can show a page-by-page compare view.
    page_previews: List[dict] = []
    done = None
    for ev in parse_pdf_iter(path, mode, max_pages, judge):
        if ev["type"] == "page":
            page_md.append(f"\n\n<!-- Page {ev['page']} ({ev['mode']}) -->\n\n{ev['markdown']}")
            images.extend(ev.get("figures", []))
            blocks.extend(ev.get("blocks", []))
            page_previews.append({
                "page": ev["page"],
                "mode": ev["mode"],
                "markdown": ev["markdown"],
                "image_url": ev.get("page_image_url"),
                "figures_count": len(ev.get("figures") or []),
                "blocks_count": len(ev.get("blocks") or []),
            })
        elif ev["type"] == "done":
            done = ev
    return {
        "pages": done["pages"],
        "markdown": "\n".join(page_md).strip(),
        "images": images,
        "blocks": blocks,
        "page_previews": page_previews,
        "timing": done["timing"],
        "judge_avg": done.get("judge_avg"),
        "cost": done["cost"],
    }
