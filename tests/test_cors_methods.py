"""
Guards the CORS allow_methods list against the app's own routing table.

Why this exists: message feedback and conversation rename shipped as PATCH
routes while CORSMiddleware's allow_methods listed only GET/POST/PUT/DELETE/
OPTIONS. Every server-side test passed — pytest and `requests` never send a
CORS preflight, because CORS is enforced by the *browser*, not the server —
so the failure only appeared in a real browser: preflight 400, the fetch never
fires, and the UI silently rolls back its optimistic update.

Deriving the expected verbs from app.routes rather than hardcoding them means
adding a route with a new method fails here instead of in production.
"""
import pytest
from starlette.middleware.cors import CORSMiddleware

from main import app


def _cors_allowed_methods():
    for mw in app.user_middleware:
        if mw.cls is CORSMiddleware:
            kwargs = getattr(mw, "kwargs", {}) or {}
            return set(kwargs.get("allow_methods") or [])
    pytest.fail("CORSMiddleware is not installed on the app")


def _methods_used_by_routes():
    """Every HTTP verb the app actually serves.

    Must recurse: this FastAPI version keeps included routers as nested
    _IncludedRouter objects rather than flattening them into app.routes, so a
    non-recursive walk sees only the four built-in /docs-style GET routes and
    the assertion below passes no matter what — which is exactly how a
    first draft of this test silently proved nothing.
    """
    used = set()

    seen = set()

    def walk(routes):
        for route in routes:
            if id(route) in seen:
                continue
            seen.add(id(route))
            for m in (getattr(route, "methods", None) or set()):
                used.add(m.upper())
            # fastapi's _IncludedRouter exposes its real router here; it has
            # neither .routes nor .methods of its own.
            inner = getattr(route, "original_router", None)
            if inner is not None:
                walk(getattr(inner, "routes", []) or [])
            nested = getattr(route, "routes", None)
            if nested:
                walk(nested)

    walk(app.routes)

    # Second, independent source: the generated OpenAPI schema. Belt and
    # braces, because the route-walk above depends on FastAPI internals that
    # have already changed shape once.
    try:
        for _path, ops in (app.openapi().get("paths") or {}).items():
            for verb in ops:
                if verb.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
                    used.add(verb.upper())
    except Exception:
        pass

    # HEAD is added implicitly by Starlette alongside GET and is never
    # preflighted; OPTIONS is the preflight itself.
    return used - {"HEAD"}


def test_every_routed_method_is_cors_allowed():
    allowed = _cors_allowed_methods()
    if "*" in allowed:
        return  # wildcard covers everything

    missing = _methods_used_by_routes() - allowed
    assert not missing, (
        f"These HTTP methods are used by real routes but are NOT in "
        f"CORSMiddleware allow_methods: {sorted(missing)}. A browser's "
        f"preflight will fail with 400 and the request will never reach the "
        f"app, even though server-side tests pass. Add them in main.py."
    )


def test_patch_specifically_is_allowed():
    """Pinned because the two features that regressed both use PATCH."""
    allowed = _cors_allowed_methods()
    assert "*" in allowed or "PATCH" in allowed, (
        "PATCH missing from CORS allow_methods — message feedback "
        "(thumbs up/down) and conversation rename both use it"
    )
