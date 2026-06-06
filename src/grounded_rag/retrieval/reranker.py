"""Cross-encoder reranker — Phase 4.

Candidates: bge-reranker-v2-m3 (local), Qwen3-Reranker (local), Cohere Rerank (API).
Cross-encoders jointly encode (query, passage) pairs and produce a relevance score
that is much more accurate than bi-encoder cosine similarity.
"""
from __future__ import annotations

# TODO Phase 4: implement cross-encoder reranker
