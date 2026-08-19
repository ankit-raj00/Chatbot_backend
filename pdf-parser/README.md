# PDF Parser Service

A standalone microservice that turns **scanned / image-only PDFs** into clean Markdown for RAG
ingestion — the case ordinary text extraction cannot handle at all, because there is no text layer
to extract.

It runs on its own host, not inside the main backend. The backend calls it over HTTP
(`rag/parsers/parser_client.py`), and falls back to LlamaParse if it is unavailable.

---

## Why it exists

`PyMuPDF` and similar extractors read the text layer of a PDF. A scanned document has no text
layer — every page is a photograph — so those tools return empty strings. The usual answer is OCR,
but flat OCR loses the thing that matters most for retrieval: **structure**. A table becomes a
soup of numbers, a two-column page interleaves into nonsense, and a figure caption detaches from
its figure. Chunks built from that retrieve badly.

This service keeps the structure by detecting *regions* first and transcribing each one in
isolation, so the model is never asked to read a whole cluttered page at once.

## Pipeline

```
PyMuPDF render → DocLayout-YOLO region detection → crop
      → per-region VLM transcription (Gemini via OmniRoute)
      → whole-page structure pass → merged Markdown
```

1. **Render** the page to an image at a configurable DPI (`RENDER_DPI`, default 180).
2. **Detect regions** with DocLayout-YOLO — text blocks, tables, figures, titles.
3. **Crop and transcribe each region separately.** A tight crop is a much easier prompt than a
   full page, so tables and multi-column layouts survive.
4. **Structure pass** over the whole page to fix reading order, which per-region calls cannot know
   on their own.
5. **Emit** Markdown, plus extracted figure images (base64, optionally uploaded to Cloudinary),
   with per-run timing and token cost.

## Modes

Selectable per request, so the same PDF can be compared on cost and quality:

| Mode | What it does | Cost |
|---|---|---|
| `digital` | PyMuPDF text extraction only | ~free, no model call |
| `wholepage` | Render page → one VLM call | cheapest model path |
| `region` | DocLayout-YOLO → crop → VLM per region | highest quality, most calls |
| `auto` | Per page: `digital` if it has a real text layer, else `region` | recommended |

`auto` matters on real documents, which are usually mixed — a born-digital report with a few
scanned appendix pages should not pay for vision on every page.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + active vision model |
| `POST /parse` | Parse a PDF, return the full result |
| `POST /parse/stream` | Same, streaming per-page progress |
| `GET /dashboard` | Web UI: run history, per-page output, timing and cost |

The dashboard is self-contained — its own login and signed session cookie, no dependency on the
main app's auth. It is a debugging and cost-comparison tool: every run is recorded, so a
regression in output quality or a jump in cost per page is visible rather than inferred.

## Configuration

All configuration is environment-based; there are no credentials in this source.

| Variable | Purpose |
|---|---|
| `OMNIROUTE_BASE_URL` | LLM gateway the vision calls go through |
| `VISION_MODEL` / `STRUCTURE_MODEL` | Model for region transcription / reading-order pass |
| `YOLO_WEIGHTS` | Path to the DocLayout-YOLO weights |
| `RENDER_DPI` | Page render resolution |
| `REGION_WORKERS` | Region-call concurrency — **deliberately defaults to 1** |
| `STRUCTURE_PASS` | Toggle the reading-order pass |
| `MONGO_URI` / `MONGO_DB_NAME` | Run history for the dashboard |
| `CLOUDINARY_*` | Optional durable storage for extracted figures |
| `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD` | Dashboard login. With no password set, login returns 503 rather than allowing access |

`REGION_WORKERS=1` is not an oversight: parallel region calls tripped the gateway's rate limits, so
the sequential default is what actually completes. A heavily illustrated scan can therefore take
minutes, which is why the caller uses a long timeout and a streaming endpoint.

## Running it

```bash
docker build -t pdf-parser .
docker run -p 8000:8000 --env-file .env -v /models:/models pdf-parser
```

The DocLayout-YOLO weights are mounted rather than baked into the image, so the model can be
swapped without a rebuild.
