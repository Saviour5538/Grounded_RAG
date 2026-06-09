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
    """Sparse BM25 retriever backed by either Qdrant or any vector store.

    Pass either:
      - qdrant_client + collection  (Qdrant mode, scrolls via Qdrant API)
      - scroll_fn                   (Pinecone / any store mode — callable returning
                                     list[dict] of all chunks)

    The BM25 index is built lazily on the first retrieve() call and cached.
    """

    def __init__(
        self,
        qdrant_client: Any = None,
        collection: str = "",
        scroll_fn: Any = None,
    ):
        self.client = qdrant_client
        self.collection = collection
        self._scroll_fn = scroll_fn
        self._corpus: list[dict[str, Any]] = []
        self._bm25: Any = None

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi

        t0 = time.time()

        if self._scroll_fn is not None:
            self._corpus = self._scroll_fn()
        else:
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

        if not self._corpus:
            logger.warning("BM25: corpus is empty — sparse retrieval will be skipped until index is populated")
            return

        tokenized = [_tokenize(c["text"]) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "BM25 index ready: %d chunks in %.1fs", len(self._corpus), time.time() - t0
        )

    def invalidate(self) -> None:
        """Drop the cached BM25 index so it rebuilds on the next retrieve() call.

        Call this after new chunks are indexed so the BM25 corpus stays in sync
        with the vector store.
        """
        self._corpus = []
        self._bm25 = None
        logger.info("BM25 index invalidated — will rebuild on next query")

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Return top_k chunks ranked by BM25 score for the query."""
        if self._bm25 is None:
            self._build_index()
        if self._bm25 is None:
            return []  # corpus empty — dense-only fallback

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
