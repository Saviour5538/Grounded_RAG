"""Hash-based document deduplication.

Uses a SHA-256 fingerprint of the lowercased, stripped text so the same content
from different sources is recognised as a duplicate.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator

from grounded_rag.ingest.loader import Document


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def dedup_documents(docs: Iterator[Document]) -> Iterator[Document]:
    """Yield unique documents, silently dropping exact-text duplicates."""
    seen: set[str] = set()
    for doc in docs:
        fp = _fingerprint(doc.text)
        if fp not in seen:
            seen.add(fp)
            yield doc
