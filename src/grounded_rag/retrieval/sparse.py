"""BM25 sparse retrieval over the Qdrant chunk corpus.

Builds an in-memory BM25Okapi index by scrolling all chunks from Qdrant on the
first retrieve() call, then caches the index for subsequent queries.
No local ML models — pure keyword frequency statistics (rank_bm25 package).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Sparse BM25 retriever backed by the Qdrant chunk corpus.

    The BM25 index is built lazily on the first retrieve() call by scrolling all
    stored points from the Qdrant collection. Subsequent calls use the cached index.
    """

    def __init__(self, qdrant_client: Any, collection: str):
        self.client = qdrant_client
        self.collection = collection
        self._corpus: list[dict[str, Any]] = []
        self._bm25: Any = None

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi

        t0 = time.time()
        logger.info("Building BM25 index — scrolling Qdrant collection '%s'…", self.collection)

        offset = None
        while True:
            result, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=True,
                offset=offset,
            )
            for point in result:
                p = point.payload
                self._corpus.append({
                    "chunk_id":    p["chunk_id"],
                    "doc_id":      p["doc_id"],
                    "text":        p["text"],
                    "source":      p.get("source", ""),
                    "score":       0.0,
                    "chunk_index": p.get("chunk_index", 0),
                    "metadata": {
                        k: v for k, v in p.items()
                        if k not in {"chunk_id", "doc_id", "text", "source", "chunk_index"}
                    },
                })
            if offset is None:
                break

        tokenized = [_tokenize(c["text"]) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "BM25 index ready: %d chunks in %.1fs", len(self._corpus), time.time() - t0
        )

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Return top_k chunks ranked by BM25 score for the query."""
        if self._bm25 is None:
            self._build_index()

        import numpy as np

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = sorted(top_indices, key=lambda i: scores[i], reverse=True)

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self._corpus[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results
