"""Recursive character-based text chunker.

Tries paragraph → sentence → word → character splits in order, stopping as soon
as chunks fit within chunk_size. Overlap is applied by re-including the tail of
the previous chunk at the start of the next.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

from grounded_rag.ingest.loader import Document

# Separators tried in order from coarsest to finest.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A text chunk derived from a Document, ready for embedding."""

    id: str
    doc_id: str
    text: str
    chunk_index: int
    source: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def make(cls, text: str, doc_id: str, chunk_index: int, source: str, metadata: dict) -> "Chunk":
        chunk_id = hashlib.sha256(f"{doc_id}:{chunk_index}:{text[:32]}".encode()).hexdigest()[:16]
        return cls(id=chunk_id, doc_id=doc_id, text=text,
                   chunk_index=chunk_index, source=source, metadata=metadata)


# ── Core split logic ──────────────────────────────────────────────────────────

def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text to fit within chunk_size characters."""
    if separators is None:
        separators = _SEPARATORS

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep, rest_seps = separators[0], separators[1:]

    if sep == "":
        # Hard split with overlap as last resort.
        parts: list[str] = []
        start = 0
        while start < len(text):
            parts.append(text[start : start + chunk_size])
            start += chunk_size - chunk_overlap
        return [p for p in parts if p.strip()]

    segments = [s for s in text.split(sep) if s.strip()]
    chunks: list[str] = []
    current = ""

    for seg in segments:
        candidate = (current + sep + seg).strip() if current else seg
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                if len(current) > chunk_size and rest_seps:
                    chunks.extend(_split_text(current, chunk_size, chunk_overlap, rest_seps))
                else:
                    chunks.append(current.strip())
            # Start next chunk with overlap from tail of previous chunk.
            tail = current[-chunk_overlap:].strip() if chunk_overlap > 0 and current else ""
            current = (tail + sep + seg).strip() if tail else seg

    if current.strip():
        if len(current) > chunk_size and rest_seps:
            chunks.extend(_split_text(current, chunk_size, chunk_overlap, rest_seps))
        else:
            chunks.append(current.strip())

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(doc: Document, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Split one Document into a list of Chunks."""
    texts = _split_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        Chunk.make(
            text=t,
            doc_id=doc.id,
            chunk_index=i,
            source=doc.source,
            metadata=doc.metadata.copy(),
        )
        for i, t in enumerate(texts)
    ]


def chunk_documents(
    docs: Iterator[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Iterator[list[Chunk]]:
    """Yield a list of chunks for each document in the iterator."""
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if chunks:
            yield chunks
