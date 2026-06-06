"""Metadata extraction and normalization for documents."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from grounded_rag.ingest.loader import Document


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coerce all metadata values to JSON-serializable primitives."""
    result: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            result[k] = v
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, (list, tuple)):
            result[k] = [str(i) for i in v]
        else:
            result[k] = str(v)
    return result


def enrich_document(doc: Document) -> Document:
    """Add derived metadata (char/word counts, ingest timestamp) to a document."""
    meta = normalize_metadata(doc.metadata)
    meta.setdefault("char_count", len(doc.text))
    meta.setdefault("word_count", len(doc.text.split()))
    meta.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())
    return doc.model_copy(update={"metadata": meta})
