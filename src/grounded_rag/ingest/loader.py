"""Load raw documents from disk into the Document schema.

Supports JSONL, CSV, and plain-text files. The Document id is a short SHA-256
fingerprint of the text content — stable across re-ingests unless the text changes.
"""
from __future__ import annotations

import csv
import hashlib
import json
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
            text = next((obj[k] for k in _TEXT_FIELDS if obj.get(k)), "")
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
            text = row.get(text_col, "").strip()
            if not text:
                continue
            raw_id = row.get("id", "")
            doc_id = str(raw_id) if raw_id else hashlib.sha256(text.encode()).hexdigest()[:16]
            metadata = {k: v for k, v in row.items() if k not in {text_col, "id"}}
            yield Document(id=doc_id, text=text, source=str(path), metadata=metadata)


# ── Directory ─────────────────────────────────────────────────────────────────

def load_directory(directory: Path) -> Iterator[Document]:
    """Recursively load .jsonl, .json, .csv, and .txt files from a directory."""
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
            text = path.read_text(encoding="utf-8").strip()
            if text:
                yield Document.from_text(text, source=str(path))
