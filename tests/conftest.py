"""
conftest.py — global pytest configuration for ALL test files.
Forces UTF-8 output so emoji/arrow chars in structlog don't crash on Windows cp1252.
"""
import os
import sys
import io

# ── Force UTF-8 console output (fixes Windows cp1252 UnicodeEncodeError) ──────
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Rewrap stdout/stderr so emoji in structlog/print doesn't crash on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
