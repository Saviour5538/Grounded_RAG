"""Source confidence scoring and abstention gate — Phase 5.

Combines three signals into a single [0, 1] confidence score:
  1. reranker_score  — Gemini cross-encoder relevance (0-1), most reliable
  2. rrf_signal      — normalised Reciprocal Rank Fusion score
  3. token_overlap   — Jaccard overlap between query tokens and chunk tokens

If confidence < threshold the pipeline returns "Insufficient evidence found."
instead of calling the LLM — the crown-jewel abstention feature.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_MAX_RRF = 2.0 / 61.0


def _token_overlap(query: str, chunk: str) -> float:
    """Recall-based overlap: fraction of query tokens that appear in the chunk.

    Using recall (not Jaccard) because short queries like "BART" should score
    high if the chunk mentions BART, not low because the chunk has 200 other tokens.
    """
    tq = set(_TOKEN_RE.findall(query.lower()))
    tc = set(_TOKEN_RE.findall(chunk.lower()))
    if not tq:
        return 0.0
    return len(tq & tc) / len(tq)


class ConfidenceScorer:
    """Compute a retrieval confidence score and decide whether to abstain.

    Parameters
    ----------
    threshold:
        Confidence below this value triggers abstention. Default 0.3 means
        the average reranker score must be at least 3/10 for the pipeline to
        attempt an answer.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def compute(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return a dict with score (float), signals (dict), and abstain (bool).

        threshold: per-request override. If None, falls back to self.threshold
                   (set from CONFIDENCE_THRESHOLD env var / slider default).
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        if not chunks:
            return {"score": 0.0, "signals": {}, "abstain": True}

        top = chunks[:5]

        # Signal 1: reranker score (most discriminating)
        rs = [c.get("reranker_score") for c in top]
        has_reranker = all(r is not None for r in rs)
        reranker_signal = sum(rs) / len(rs) if has_reranker else None

        # Signal 2: normalised RRF score
        rrf_vals = [c.get("score", 0.0) for c in top]
        rrf_signal = min(sum(rrf_vals) / len(rrf_vals) / _MAX_RRF, 1.0)

        # Signal 3: token overlap
        overlap_signal = sum(_token_overlap(query, c["text"]) for c in top) / len(top)

        if has_reranker:
            score = 0.55 * reranker_signal + 0.25 * rrf_signal + 0.20 * overlap_signal
        else:
            score = 0.60 * rrf_signal + 0.40 * overlap_signal

        score = round(min(max(score, 0.0), 1.0), 4)

        return {
            "score": score,
            "signals": {
                "reranker_score": round(reranker_signal, 4) if has_reranker else None,
                "rrf_signal":     round(rrf_signal, 4),
                "token_overlap":  round(overlap_signal, 4),
            },
            "abstain": score < effective_threshold,
        }
