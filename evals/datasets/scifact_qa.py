"""Load SciFact queries as evaluation samples.

SciFact claims are scientific assertions (e.g. "Smoking causes lung cancer.").
We treat each claim as a query to the RAG pipeline and measure whether the
pipeline retrieves relevant context and produces a grounded answer.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_QUERIES_PATH = Path("data/processed/scifact_queries.jsonl")


def load_eval_samples(n: int = 100, seed: int = 42) -> list[dict]:
    """Return up to n randomly sampled SciFact claims as eval dicts.

    Each dict has:
        question  str  — the claim text (used as the RAG query)
        claim_id  str  — original SciFact claim id
    """
    if not _QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Queries file not found at {_QUERIES_PATH}. "
            "Run `python scripts/ingest_corpus.py` first."
        )

    samples: list[dict] = []
    with _QUERIES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("query") or obj.get("question", "")
            if not text:
                continue
            samples.append({"question": text, "claim_id": str(obj.get("id", ""))})

    random.seed(seed)
    random.shuffle(samples)
    return samples[:n]
