"""In-memory LRU query cache — Phase 7.

Caches full pipeline results keyed by a normalised query hash so repeated
questions return instantly without touching Qdrant or the LLM.

Latency impact is recorded in each cached response so the eval harness can
report cache-hit vs cold-path latency separately.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


def _key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


class QueryCache:
    """LRU in-memory cache for pipeline results.

    Parameters
    ----------
    max_size:
        Maximum number of entries; oldest entry is evicted when exceeded.
    """

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> dict[str, Any] | None:
        k = _key(query)
        if k in self._store:
            self._store.move_to_end(k)
            self.hits += 1
            logger.debug("Cache HIT  key=%s query=%r", k, query[:60])
            return dict(self._store[k])  # shallow copy so caller can mutate
        self.misses += 1
        return None

    def set(self, query: str, result: dict[str, Any]) -> None:
        k = _key(query)
        self._store[k] = result
        self._store.move_to_end(k)
        if len(self._store) > self.max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
