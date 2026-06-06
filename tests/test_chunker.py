"""Unit tests for the recursive text chunker."""
import pytest

from grounded_rag.ingest.chunker import Chunk, _split_text, chunk_document
from grounded_rag.ingest.loader import Document


# ── _split_text ───────────────────────────────────────────────────────────────

def test_short_text_returned_as_single_chunk():
    parts = _split_text("Hello world.", chunk_size=512, chunk_overlap=64)
    assert parts == ["Hello world."]


def test_empty_text_returns_empty():
    assert _split_text("", chunk_size=512, chunk_overlap=0) == []
    assert _split_text("   ", chunk_size=512, chunk_overlap=0) == []


def test_paragraph_split():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    parts = _split_text(text, chunk_size=30, chunk_overlap=0)
    assert len(parts) >= 2
    assert all(len(p) > 0 for p in parts)


def test_no_chunk_exceeds_size():
    text = "word " * 200
    parts = _split_text(text, chunk_size=64, chunk_overlap=8)
    for p in parts:
        assert len(p) <= 64, f"Chunk too long: {len(p)} chars"


def test_hard_split_fallback():
    # A single long word with no whitespace forces the hard (character) split.
    text = "A" * 300
    parts = _split_text(text, chunk_size=100, chunk_overlap=0)
    assert len(parts) == 3
    assert all(len(p) <= 100 for p in parts)


# ── chunk_document ────────────────────────────────────────────────────────────

def test_chunk_document_produces_multiple_chunks():
    doc = Document(id="doc1", text="Hello world. " * 100)
    chunks = chunk_document(doc, chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 1


def test_all_chunks_have_correct_doc_id():
    doc = Document(id="my_doc", text="Some text. " * 60)
    chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
    assert all(c.doc_id == "my_doc" for c in chunks)


def test_chunk_metadata_is_copied():
    doc = Document(id="d1", text="Text. " * 60, metadata={"author": "Alice"})
    chunks = chunk_document(doc, chunk_size=64, chunk_overlap=0)
    assert all(c.metadata.get("author") == "Alice" for c in chunks)


def test_chunk_indices_are_sequential():
    doc = Document(id="d2", text="word " * 200)
    chunks = chunk_document(doc, chunk_size=64, chunk_overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    doc = Document(id="d3", text="word " * 200)
    chunks = chunk_document(doc, chunk_size=64, chunk_overlap=0)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_short_document_becomes_one_chunk():
    doc = Document(id="short", text="Just a short sentence.")
    chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "Just a short sentence."
