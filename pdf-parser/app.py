"""FastAPI wrapper for the hybrid PDF parser (runs as a Docker microservice)."""
import os
import json
import time
import hmac
import base64
import hashlib
import secrets
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse

from parser import parse_pdf, parse_pdf_iter
import history

app = FastAPI(title="PDF Parser Service", version="0.1.0")

DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
LOGIN_HTML = (Path(__file__).parent / "login.html").read_text(encoding="utf-8")

DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
# Derived from the password by default so no extra required env var — set
# DASHBOARD_SECRET explicitly to rotate the signing key independently of
# the login password.
DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET") or hashlib.sha256(DASHBOARD_PASSWORD.encode()).hexdigest()
SESSION_COOKIE = "dash_session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 days

# Plain HTTP Basic Auth kept re-prompting on every refresh in Chrome (browsers
# don't reliably cache Basic Auth creds on non-HTTPS origins). A real login
# form + signed session cookie is more reliable and still fully owned by this
# service — no dependency on any caller's login system.


def _make_session_token(username: str) -> str:
    expiry = int(time.time()) + SESSION_MAX_AGE
    payload = f"{username}:{expiry}"
    sig = hmac.new(DASHBOARD_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()


def _verify_session_token(token: str) -> bool:
    try:
        username, expiry, sig = base64.urlsafe_b64decode(token.encode()).decode().split(":", 2)
        expected_sig = hmac.new(DASHBOARD_SECRET.encode(), f"{username}:{expiry}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return False
        if int(expiry) < time.time():
            return False
        return secrets.compare_digest(username, DASHBOARD_USERNAME)
    except Exception:
        return False


def require_dashboard_session(request: Request) -> bool:
    """Used by the JSON /dashboard/api/* routes — 401 on failure (the page
    routes below handle their own redirect-to-login instead)."""
    if not DASHBOARD_PASSWORD:
        raise HTTPException(503, "Dashboard not configured: set DASHBOARD_PASSWORD")
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token or not _verify_session_token(token):
        raise HTTPException(401, "Not authenticated")
    return True


@app.on_event("startup")
async def _startup():
    history.init_db()


@app.get("/health")
async def health():
    return {"status": "ok", "vision_model": os.getenv("VISION_MODEL")}


@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    mode: str = Form("auto"),        # auto | digital | wholepage | region
    max_pages: int = Form(0),        # 0 = all
    judge: bool = Form(False),       # LLM-as-judge scoring
    include_b64: bool = Form(True),  # ingestion sets false (URL only, light payload)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(parse_pdf, tmp_path, mode, max_pages, judge, include_b64)
        result["filename"] = file.filename
        result["bytes"] = len(data)
        blocks = result.get("blocks") or []
        run_id = await history.record_run(
            filename=file.filename, status="success" if blocks else "empty", mode=mode,
            pages=result.get("pages"), blocks_count=len(blocks),
            figures_count=len(result.get("images") or []), bytes_size=len(data),
            duration_ms=int((time.perf_counter() - t0) * 1000), markdown=result.get("markdown"),
        )
        await history.record_pages(run_id, result.get("page_previews") or [])
        return result
    except Exception as e:
        await history.record_run(
            filename=file.filename, status="failed", mode=mode, bytes_size=len(data),
            duration_ms=int((time.perf_counter() - t0) * 1000), error=str(e),
        )
        raise HTTPException(500, f"Parse failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/parse/stream")
async def parse_stream(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    max_pages: int = Form(0),
    judge: bool = Form(False),
):
    """Stream one SSE event per page as it finishes (live status + preview)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    async def gen():
        t0 = time.perf_counter()
        pages_seen = 0
        blocks_total = 0
        figures_total = 0
        md_parts = []
        page_previews = []
        errored = None
        try:
            # parse_pdf_iter is a sync generator (torch/VLM) → run in threadpool
            async for ev in iterate_in_threadpool(parse_pdf_iter(tmp_path, mode, max_pages, judge)):
                if ev.get("type") == "page":
                    pages_seen += 1
                    blocks_total += len(ev.get("blocks") or [])
                    figures_total += len(ev.get("figures") or [])
                    if ev.get("markdown"):
                        md_parts.append(ev["markdown"])
                    page_previews.append({
                        "page": ev.get("page"), "mode": ev.get("mode"),
                        "markdown": ev.get("markdown"), "image_url": ev.get("page_image_url"),
                        "figures_count": len(ev.get("figures") or []),
                        "blocks_count": len(ev.get("blocks") or []),
                    })
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            errored = str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            run_id = await history.record_run(
                filename=file.filename,
                status="failed" if errored else ("success" if blocks_total else "empty"),
                mode=mode, pages=pages_seen or None, blocks_count=blocks_total,
                figures_count=figures_total, bytes_size=len(data),
                duration_ms=int((time.perf_counter() - t0) * 1000), error=errored,
                markdown="\n".join(md_parts) if md_parts else None,
            )
            await history.record_pages(run_id, page_previews)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


# ─── Dashboard ─────────────────────────────────────────────────────────────
# Served directly by this microservice, gated by its own login + session
# cookie (above) — not the shared nginx Bearer-token key (browsers can't
# attach that on a normal page load) and not any caller's login system.

# Both HTML pages below get re-fetched (never browser-cached) — this bit us
# once already: an old dashboard.html stayed cached in the browser across a
# deploy, silently hiding new features (page-compare) behind stale HTML.
NO_CACHE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate"}


@app.get("/dashboard/login", response_class=HTMLResponse)
async def login_page(error: str = ""):
    err_html = "<div class='err'>Invalid username or password.</div>" if error else ""
    return HTMLResponse(LOGIN_HTML.replace("{{ERROR}}", err_html), headers=NO_CACHE_HEADERS)


@app.post("/dashboard/login")
async def login_submit(username: str = Form(...), password: str = Form(...)):
    if not DASHBOARD_PASSWORD:
        raise HTTPException(503, "Dashboard not configured: set DASHBOARD_PASSWORD")
    if secrets.compare_digest(username, DASHBOARD_USERNAME) and secrets.compare_digest(password, DASHBOARD_PASSWORD):
        resp = RedirectResponse("/dashboard", status_code=303)
        resp.set_cookie(
            SESSION_COOKIE, _make_session_token(username),
            httponly=True, samesite="lax", max_age=SESSION_MAX_AGE,
        )
        return resp
    return RedirectResponse("/dashboard/login?error=1", status_code=303)


@app.get("/dashboard/logout")
async def logout():
    resp = RedirectResponse("/dashboard/login")
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token or not _verify_session_token(token):
        return RedirectResponse("/dashboard/login")
    return HTMLResponse(DASHBOARD_HTML, headers=NO_CACHE_HEADERS)


@app.get("/dashboard/api/stats")
async def dashboard_stats(_: bool = Depends(require_dashboard_session)):
    return await history.get_stats()


@app.get("/dashboard/api/runs")
async def dashboard_runs(
    limit: int = 50,
    skip: int = 0,
    status: str | None = None,
    _: bool = Depends(require_dashboard_session),
):
    return {"runs": await history.list_runs(limit=limit, skip=skip, status=status)}


@app.get("/dashboard/api/runs/{run_id}")
async def dashboard_run_detail(run_id: str, _: bool = Depends(require_dashboard_session)):
    run = await history.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/dashboard/api/runs/{run_id}/pages")
async def dashboard_run_pages(run_id: str, _: bool = Depends(require_dashboard_session)):
    return {"pages": await history.list_pages(run_id)}


# Dedicated per-run URL (real page, not just a JS overlay) — bookmarkable,
# shareable, browser back/forward works. Registered LAST among /dashboard/*
# routes: FastAPI matches routes in registration order, and this single-
# path-segment pattern would otherwise shadow /dashboard/login etc.
@app.get("/dashboard/{run_id}", response_class=HTMLResponse)
async def dashboard_run_page(run_id: str, request: Request):
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token or not _verify_session_token(token):
        return RedirectResponse("/dashboard/login")
    return HTMLResponse(DASHBOARD_HTML, headers=NO_CACHE_HEADERS)
