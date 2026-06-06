"""Unit tests for the ingest pipeline (loader, dedup, metadata)."""
import json
import tempfile
from pathlib import Path

from grounded_rag.ingest.dedup import dedup_documents
from grounded_rag.ingest.loader import Document, load_csv, load_jsonl, load_directory
from grounded_rag.ingest.metadata import enrich_document, normalize_metadata


# ── loader ────────────────────────────────────────────────────────────────────

def test_load_jsonl(tmp_path):
    data = [
        {"id": "1", "text": "First document."},
        {"id": "2", "text": "Second document.", "title": "T2"},
    ]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
    docs = list(load_jsonl(path))
    assert len(docs) == 2
    assert docs[0].id == "1"
    assert docs[0].text == "First document."
    assert docs[1].metadata.get("title") == "T2"


def test_load_jsonl_skips_empty_text(tmp_path):
    data = [{"id": "1", "text": ""}, {"id": "2", "text": "Valid."}]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
    docs = list(load_jsonl(path))
    assert len(docs) == 1
    assert docs[0].id == "2"


def test_load_csv(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("id,text,category\n1,Hello world,A\n2,Foo bar,B\n", encoding="utf-8")
    docs = list(load_csv(path))
    assert len(docs) == 2
    assert docs[0].text == "Hello world"
    assert docs[0].metadata["category"] == "A"


def test_load_directory_mixed(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"id": "1", "text": "From JSONL."}) + "\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("From text file.", encoding="utf-8")
    docs = list(load_directory(tmp_path))
    texts = {d.text for d in docs}
    assert "From JSONL." in texts
    assert "From text file." in texts


# ── dedup ─────────────────────────────────────────────────────────────────────

def test_dedup_drops_exact_duplicates():
    docs = [
        Document(id="a", text="Same text"),
        Document(id="b", text="Same text"),
        Document(id="c", text="Different text"),
    ]
    result = list(dedup_documents(iter(docs)))
    assert len(result) == 2


def test_dedup_case_insensitive():
    docs = [
        Document(id="a", text="HELLO WORLD"),
        Document(id="b", text="hello world"),
    ]
    result = list(dedup_documents(iter(docs)))
    assert len(result) == 1


# ── metadata ──────────────────────────────────────────────────────────────────

def test_enrich_document_adds_word_count():
    doc = Document(id="x", text="one two three")
    enriched = enrich_document(doc)
    assert enriched.metadata["word_count"] == 3
    assert enriched.metadata["char_count"] == len("one two three")


def test_normalize_metadata_coerces_types():
    from datetime import datetime
    meta = {"num": 42, "flag": True, "dt": datetime(2024, 1, 1), "lst": [1, 2]}
    result = normalize_metadata(meta)
    assert result["num"] == 42
    assert result["flag"] is True
    assert result["dt"] == "2024-01-01T00:00:00"
    assert result["lst"] == ["1", "2"]
