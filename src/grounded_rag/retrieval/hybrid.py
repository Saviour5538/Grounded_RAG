"""Hybrid retrieval: BM25 (sparse) + dense vectors, fused with Reciprocal Rank Fusion.

RRF score(d) = Σ_r  1 / (k + rank_r(d))
k=60 is the standard from the original RRF paper (Cormack et al., 2009).
Higher-ranked results from either retriever get more weight; results that appear
in both lists are boosted, which is the key advantage over single-mode retrieval.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from typing import Any as _Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses dense (Qdrant) and BM25 results via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: _Any,
        sparse: _Any,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_k: int = 50,
        sparse_k: int = 50,
        dense_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return top_k chunks, fused from dense + BM25 results via RRF.

        dense_query: optional alternative text for the dense retriever (used by HyDE —
                     a hypothetical answer that embeds closer to relevant docs than
                     the raw keyword query). BM25 always uses the original query.
        """
        _dense_q = dense_query or query
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_dense  = pool.submit(self.dense.retrieve,  _dense_q, top_k=dense_k)
            fut_sparse = pool.submit(self.sparse.retrieve, query,    top_k=sparse_k)
            dense_hits  = fut_dense.result()
            sparse_hits = fut_sparse.result()
        logger.debug(
            "RRF input — dense: %d hits, sparse: %d hits", len(dense_hits), len(sparse_hits)
        )
        return self._rrf_fuse(dense_hits, sparse_hits, top_k)

    def _rrf_fuse(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
    ) -> list[dict[str, Any]]:
        rrf_scores: dict[str, float] = {}
        chunk_store: dict[str, dict] = {}

        for rank, chunk in enumerate(dense):
            key = chunk["chunk_id"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_store[key] = chunk

        for rank, chunk in enumerate(sparse):
            key = chunk["chunk_id"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in chunk_store:
                chunk_store[key] = chunk

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for key, score in ranked:
            chunk = dict(chunk_store[key])
            chunk["score"] = round(score, 6)
            results.append(chunk)
        return results

    def corpus_stats(self) -> dict[str, Any]:
        """Return stats about the indexed corpus for the browser UI."""
        try:
            total = self.dense.collection_size()
            sources: dict[str, int] = {}
            if self.sparse._corpus:
                for chunk in self.sparse._corpus:
                    src = chunk.get("source", "unknown") or "unknown"
                    sources[src] = sources.get(src, 0) + 1
            return {
                "total_chunks":   total,
                "unique_sources": len(sources),
                "bm25_built":     self.sparse._bm25 is not None,
                "sources": [
                    {"name": k, "chunks": v}
                    for k, v in sorted(sources.items(), key=lambda x: -x[1])
                ],
            }
        except Exception as exc:
            logger.warning("corpus_stats failed: %s", exc)
            return {"total_chunks": 0, "unique_sources": 0, "bm25_built": False, "sources": [], "error": str(exc)}

    # ── Delegate index-management methods to the dense retriever ─────────────
    def index_chunks(self, chunks: list, batch_size: int = 128) -> int:
        return self.dense.index_chunks(chunks, batch_size=batch_size)

    def ensure_collection(self) -> None:
        return self.dense.ensure_collection()

    def collection_size(self) -> int:
        return self.dense.collection_size()
