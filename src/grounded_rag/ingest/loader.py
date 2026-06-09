"""Load raw documents from disk into the Document schema.

Supports JSONL, CSV, plain-text, and PDF files. The Document id is a short
SHA-256 fingerprint of the text content — stable across re-ingests unless the
text changes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel


class Document(BaseModel):
    """A single source document before chunking."""

    id: str
    text: str
    source: str = ""
    metadata: dict = {}
    created_at: datetime | None = None

    @classmethod
    def from_text(cls, text: str, source: str = "", metadata: dict | None = None) -> "Document":
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(id=doc_id, text=text, source=source, metadata=metadata or {})


# ── JSONL ─────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """NFKC unicode normalization + collapse runs of whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


# Field names tried in order when looking for the main document text.
_TEXT_FIELDS = ("text", "content", "abstract", "body", "passage")


def load_jsonl(path: Path) -> Iterator[Document]:
    """Load documents from a JSONL file. Each line must have at least one text field."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _normalize(next((obj[k] for k in _TEXT_FIELDS if obj.get(k)), ""))
            if not text:
                continue
            raw_id = obj.get("id") or obj.get("_id")
            doc_id = str(raw_id) if raw_id else hashlib.sha256(text.encode()).hexdigest()[:16]
            metadata = {k: v for k, v in obj.items() if k not in {*_TEXT_FIELDS, "id", "_id"}}
            yield Document(id=doc_id, text=text, source=str(path), metadata=metadata)


# ── CSV ───────────────────────────────────────────────────────────────────────

def load_csv(path: Path, text_col: str = "text") -> Iterator[Document]:
    """Load documents from a CSV file."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _normalize(row.get(text_col, ""))
            if not text:
                continue
            raw_id = row.get("id", "")
            doc_id = str(raw_id) if raw_id else hashlib.sha256(text.encode()).hexdigest()[:16]
            metadata = {k: v for k, v in row.items() if k not in {text_col, "id"}}
            yield Document(id=doc_id, text=text, source=str(path), metadata=metadata)


# ── PDF ───────────────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> Iterator[Document]:
    """Extract text from a PDF file and yield one Document per logical section.

    Cleans PDF-specific artifacts:
    - Repeated headers/footers (lines appearing on >30% of pages)
    - Isolated page numbers
    - Hyphenated line breaks ("connec-\\ntion" → "connection")
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("Install pymupdf to load PDF files:  pip install pymupdf")

    doc = fitz.open(str(path))
    pages_raw: list[str] = [page.get_text() for page in doc]
    doc.close()

    if not pages_raw:
        return

    # Detect boilerplate: short lines that repeat across ≥30% of pages
    threshold = max(2, len(pages_raw) * 0.30)
    line_freq: Counter[str] = Counter()
    for page_text in pages_raw:
        lines = page_text.splitlines()
        # Only sample the header/footer zones (first 3 + last 3 lines per page)
        for line in lines[:3] + lines[-3:]:
            stripped = line.strip()
            if stripped and len(stripped) < 120:
                line_freq[stripped] += 1
    boilerplate = {line for line, cnt in line_freq.items() if cnt >= threshold}

    # Strip boilerplate and join pages
    cleaned_pages: list[str] = []
    for page_text in pages_raw:
        lines = [l for l in page_text.splitlines() if l.strip() not in boilerplate]
        cleaned_pages.append("\n".join(lines))

    combined = "\n".join(cleaned_pages)

    # Fix hyphenated line breaks: "connec-\ntion" → "connection"
    combined = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", combined)

    # Remove isolated page numbers (a lone integer on its own line)
    combined = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", combined)

    text = _normalize(combined)
    if text:
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        title = path.stem.replace("_", " ").replace("-", " ").title()
        yield Document(
            id=doc_id,
            text=text,
            source=str(path),
            metadata={"title": title, "filename": path.name},
        )


# ── Directory ─────────────────────────────────────────────────────────────────

def load_directory(directory: Path) -> Iterator[Document]:
    """Recursively load .jsonl, .json, .csv, .txt, and .pdf files from a directory."""
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".json"):
            yield from load_jsonl(path)
        elif suffix == ".csv":
            yield from load_csv(path)
        elif suffix == ".txt":
            text = _normalize(path.read_text(encoding="utf-8"))
            if text:
                yield Document.from_text(text, source=str(path))
        elif suffix == ".pdf":
            yield from load_pdf(path)
