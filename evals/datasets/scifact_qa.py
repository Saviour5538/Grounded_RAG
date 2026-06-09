"""Load SciFact queries as evaluation samples.

SciFact claims are scientific assertions (e.g. "Smoking causes lung cancer.").
We treat each claim as a query to the RAG pipeline and measure whether the
pipeline retrieves relevant context and produces a grounded answer.

The `has_evidence` field comes from SciFact qrels:
  True  → claim has supporting/refuting evidence in the corpus (pipeline should answer)
  False → NOT_ENOUGH_INFO claim (pipeline should abstain — crown-jewel test)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_QUERIES_PATH = Path("data/processed/scifact_queries.jsonl")
_QRELS_PATH   = Path("data/processed/scifact_qrels.jsonl")


def _load_evidence_set() -> set[str]:
    """Return the set of claim_ids that have evidence in the corpus."""
    if not _QRELS_PATH.exists():
        return set()
    evidence: set[str] = set()
    with _QRELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("has_evidence"):
                evidence.add(str(obj["claim_id"]))
    return evidence


def load_eval_samples(n: int = 100, seed: int = 42) -> list[dict]:
    """Return up to n randomly sampled SciFact claims as eval dicts.

    Each dict has:
        question      str   — the claim text (used as the RAG query)
        claim_id      str   — original SciFact claim id
        has_evidence  bool  — True if corpus contains supporting/refuting evidence
                              False = NOT_ENOUGH_INFO (pipeline should abstain)
    """
    if not _QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Queries file not found at {_QUERIES_PATH}. "
            "Run `python scripts/ingest_corpus.py` first."
        )

    evidence_set = _load_evidence_set()
    has_qrels = bool(evidence_set)

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
            claim_id = str(obj.get("id", ""))
            samples.append({
                "question":     text,
                "claim_id":     claim_id,
                # If qrels not downloaded yet, default all to True so other
                # metrics still run — abstention_correctness is skipped.
                "has_evidence": (claim_id in evidence_set) if has_qrels else None,
            })

    random.seed(seed)
    random.shuffle(samples)
    return samples[:n]
