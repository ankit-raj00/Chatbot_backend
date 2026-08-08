FROM python:3.11-slim

# System deps: build tools (for packages needing compilation), poppler-utils
# (pdftoppm, used by the create-pdf skill), and weasyprint's rendering stack
# (cairo/pango/gdk-pixbuf) — weasyprint is in requirements.txt for PDF/doc
# generation and fails at import time without these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Dedicated low-privilege user for sandboxed run_python/run_shell subprocess
# execution (Tier 1.1, HARDENING_PLAN.md). The backend process itself still
# runs as root (needed to chown per-user/per-conversation workspace
# directories over to this user — see utils/workspace.py), but
# user-authored agent code now executes AS this UID instead of inheriting
# root, so the OS itself can finally tell "trusted app code" apart from
# "arbitrary sandboxed code" — confirmed live via red-team testing that
# with no separation at all, sandboxed code could reach the app's own
# internal ports and the cloud metadata endpoint with zero restriction.
RUN useradd --uid 1001 --create-home --shell /usr/sbin/nologin sandboxrunner
ENV SANDBOX_UID=1001
ENV SANDBOX_GID=1001

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
