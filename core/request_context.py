"""
Request context middleware — adds correlation IDs to every log entry.

Every HTTP request gets a unique request_id (UUID).
This ID is bound to structlog's contextvars, so ALL log statements
made during that request automatically include it.

WHY: When debugging a production issue, you want to filter logs to
     a single request. Without correlation IDs, logs from concurrent
     requests are interleaved and impossible to follow.
"""

import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class CorrelationIdMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        # Generate or propagate a request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind context so all log calls in this request include request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # Pass the request ID back in the response header
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response
