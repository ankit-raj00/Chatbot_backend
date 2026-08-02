"""
conftest.py — global pytest configuration for ALL test files.
Forces UTF-8 output so emoji/arrow chars in structlog don't crash on Windows cp1252.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load backend/.env FIRST so env-dependent modules (e.g. embeddings clients that
# validate GOOGLE_API_KEY at construction) have their config before any test
# imports them. conftest is imported before collection, so this wins the race.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Force UTF-8 console output (fixes Windows cp1252 UnicodeEncodeError) ──────
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Reconfigure the EXISTING stream to UTF-8 instead of REPLACING it with a new
# TextIOWrapper. Replacing the object breaks pytest's output capture (it wraps
# stdout/stderr itself), which previously forced runs to use `-s`. reconfigure()
# is a no-op / safely skipped on stream objects that don't support it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
