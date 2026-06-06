"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion — Phase 3.

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each retrieval system i.
Typical k=60 is used to dampen the impact of very high ranks.
"""
from __future__ import annotations

# TODO Phase 3: implement RRF fusion of dense + sparse results
