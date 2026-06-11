"""BM25 sparse retrieval over the chunk corpus.

Builds an in-memory BM25Okapi index by scrolling all chunks from the vector
store on the first retrieve() call.  The index is persisted to a pickle file
so subsequent server restarts load in milliseconds instead of re-scrolling.

Invalidation (e.g. after new chunks are indexed) deletes the pickle and forces
a full rebuild on the next query.
"""
from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Sparse BM25 retriever backed by either Qdrant or any vector store.

    Pass either:
      - qdrant_client + collection  (Qdrant mode, scrolls via Qdrant API)
      - scroll_fn                   (Pinecone / any store mode — callable
                                     returning list[dict] of all chunks)

    The BM25 index is built lazily on the first retrieve() call, persisted to
    cache_path, and reloaded from disk on subsequent server starts.
    """

    def __init__(
        self,
        qdrant_client: Any = None,
        collection: str = "",
        scroll_fn: Any = None,
        cache_path: str = "data/bm25_index.pkl",
    ):
        self.client = qdrant_client
        self.collection = collection
        self._scroll_fn = scroll_fn
        self._cache_path = Path(cache_path)
        self._corpus: list[dict[str, Any]] = []
        self._bm25: Any = None

    def _load_from_disk(self) -> bool:
        """Try to load index from the pickle cache. Returns True on success."""
        if not self._cache_path.exists():
            return False
        try:
            t0 = time.time()
            with self._cache_path.open("rb") as f:
                saved = pickle.load(f)
            self._corpus = saved["corpus"]
            self._bm25   = saved["bm25"]
            logger.info(
                "BM25 index loaded from disk cache: %d chunks in %.2fs",
                len(self._corpus), time.time() - t0,
            )
            return True
        except Exception as exc:
            logger.warning("BM25 disk cache corrupt or unreadable (%s) — will rebuild", exc)
            self._corpus = []
            self._bm25   = None
            return False

    def _save_to_disk(self) -> None:
        """Persist the current index to the pickle cache."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("wb") as f:
                pickle.dump({"corpus": self._corpus, "bm25": self._bm25}, f)
            logger.info("BM25 index saved to disk: %s", self._cache_path)
        except Exception as exc:
            logger.warning("BM25 cache save failed (non-fatal): %s", exc)

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi

        # Fast path — load from disk if available
        if self._load_from_disk():
            return

        # Slow path — scroll vector store and build from scratch
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
        logger.info("BM25 index built: %d chunks in %.1fs", len(self._corpus), time.time() - t0)

        self._save_to_disk()

    def invalidate(self) -> None:
        """Drop the in-memory index and delete the disk cache.

        Call this after new chunks are indexed so the next query rebuilds
        from the updated vector store.
        """
        self._corpus = []
        self._bm25   = None

        if self._cache_path.exists():
            try:
                self._cache_path.unlink()
                logger.info("BM25 disk cache deleted — will rebuild on next query")
            except Exception as exc:
                logger.warning("Could not delete BM25 cache file: %s", exc)
        else:
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
