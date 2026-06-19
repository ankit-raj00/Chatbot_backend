"""Tests for the universal file reader."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pathlib import Path
from services.universal_file_reader import extract_any_file


@pytest.mark.asyncio
async def test_read_text_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("Hello, world!\nLine 2")
        fname = f.name
    try:
        result = await extract_any_file(Path(fname))
        assert result["type"] == "text"
        assert "Hello, world!" in result["content"]
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_read_python_file():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write("def hello():\n    return 'world'\n")
        fname = f.name
    try:
        result = await extract_any_file(Path(fname))
        assert result["type"] == "text"
        assert "def hello" in result["content"]
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_read_csv_file():
    import csv
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                      delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
        writer.writeheader()
        writer.writerows([
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob",   "age": "25", "city": "LA"},
        ])
        fname = f.name
    try:
        result = await extract_any_file(Path(fname))
        assert result["type"] == "csv", f"Expected csv, got {result['type']}: {result}"
        assert result["row_count"] == 2
        assert "name" in result["columns"]
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_read_json_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        import json
        json.dump({"key": "value", "number": 42}, f)
        fname = f.name
    try:
        result = await extract_any_file(Path(fname))
        assert result["type"] == "json", f"Expected json, got {result['type']}: {result}"
        assert "value" in result["content"]
    finally:
        os.unlink(fname)


@pytest.mark.asyncio
async def test_read_markdown_file():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write("# Title\n\nSome content here.")
        fname = f.name
    try:
        result = await extract_any_file(Path(fname))
        assert result["type"] == "text"
        assert "# Title" in result["content"]
    finally:
        os.unlink(fname)
