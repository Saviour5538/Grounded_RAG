"""Source confidence scoring and abstention gate — Phase 5.

The confidence score combines:
  - reranker score (cross-encoder relevance, Phase 4)
  - BM25/dense agreement (do both systems agree on the top chunks? Phase 3)
  - freshness (recency of source document, if available)
  - query-context overlap (token overlap between query and retrieved chunks)

If the aggregate score falls below config.confidence_threshold, the pipeline
returns "Insufficient evidence found." instead of calling the LLM.

This is the crown-jewel feature: correct abstention on unanswerable questions
without hallucinating an answer.
"""
from __future__ import annotations

# TODO Phase 5: implement confidence scoring and abstention threshold
