"""Dense vector retrieval backed by Qdrant.

Phase 1: embed via Gemini or OpenAI API, upsert to Qdrant, retrieve by
cosine similarity. No local models — zero CPU load on the host machine.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from grounded_rag.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """API-only embedding wrapper — no local models, no PyTorch.

    Providers:
      "gemini" → Google text-embedding-004 via google-genai (uses task_type
                 for better retrieval: RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
      "openai" → OpenAI text-embedding-3-small via openai SDK
    """

    def __init__(self, model_name: str, provider: str = "gemini", api_key: str = ""):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self._client: Any = None

    def _load(self) -> None:
        if self._client is not None:
            return
        if self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Gemini embedding model ready: %s", self.model_name)
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
            logger.info("OpenAI embedding model ready: %s", self.model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider!r}")

    def embed(
        self,
        texts: list[str],
        batch_size: int = 20,
        is_query: bool = False,
        inter_batch_delay: float = 0.2,
    ) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalised embeddings.

        Uses batched API calls (one request per batch) with a small delay between
        batches to stay within free-tier rate limits. Retries up to 5 times with
        exponential backoff on 429 RESOURCE_EXHAUSTED errors.

        is_query=True uses RETRIEVAL_QUERY task type (Gemini only).
        """
        self._load()
        results: list[list[float]] = []

        if self.provider == "gemini":
            from google import genai as _genai
            from google.genai import types as genai_types
            task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self._embed_gemini_batch_with_retry(
                    batch, task, genai_types, _genai
                )
                results.extend([e.values for e in resp.embeddings])
                if i + batch_size < len(texts):
                    time.sleep(inter_batch_delay)
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self._client.embeddings.create(model=self.model_name, input=batch)
                results.extend([d.embedding for d in resp.data])
                if i + batch_size < len(texts):
                    time.sleep(inter_batch_delay)

        arr = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    def _embed_gemini_batch_with_retry(self, batch, task, genai_types, genai_module):
        """Call Gemini embed_content for a batch, retrying on 429 with backoff."""
        max_retries = 6
        delay = 5.0  # start with 5s on first 429
        for attempt in range(max_retries):
            try:
                return self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(task_type=task),
                )
            except Exception as exc:
                is_rate_limit = (
                    "429" in str(exc)
                    or "RESOURCE_EXHAUSTED" in str(exc)
                    or "quota" in str(exc).lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Waiting %.0fs before retry...",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise


# ── Qdrant retriever ──────────────────────────────────────────────────────────

class DenseRetriever:
    """Embed chunks, upsert to Qdrant, and retrieve top-k by cosine similarity."""

    def __init__(
        self,
        collection: str,
        embedding_model: EmbeddingModel,
        dim: int,
        qdrant_url: str = "http://localhost:6333",
        qdrant_mode: str = "local",
        qdrant_local_path: str = "./qdrant_data",
        qdrant_api_key: str = "",
    ):
        if qdrant_mode == "local":
            self.client = QdrantClient(path=qdrant_local_path)
            logger.info("Using local Qdrant storage at: %s", qdrant_local_path)
        else:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key or None,
                timeout=60,
            )
            logger.info("Connecting to Qdrant server at: %s", qdrant_url)
        self.collection = collection
        self.embedder = embedding_model
        self.dim = dim

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s' (dim=%d)", self.collection, self.dim)

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        """Embed and upsert chunks into Qdrant. Returns the number of points indexed."""
        self.ensure_collection()
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch]).tolist()
            points = [
                PointStruct(
                    id=_chunk_id_to_int(c.id),
                    vector=vec,
                    payload={
                        "chunk_id": c.id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        **c.metadata,
                    },
                )
                for c, vec in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            total += len(points)
        return total

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the top-k most similar chunks for `query`."""
        query_vec = self.embedder.embed([query], is_query=True)[0].tolist()
        # Support multiple qdrant-client versions: prefer `query_points`,
        # fall back to `search` or `search_points` depending on availability.
        hits = None
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection,
                query=query_vec,
                limit=top_k,
                with_payload=True,
            )
            # query_points returns an object with `.points`
            hits = getattr(result, "points", result)
        elif hasattr(self.client, "search"):
            # older/newer clients may expose `search`
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True,
            )
        elif hasattr(self.client, "search_points"):
            hits = self.client.search_points(
                collection_name=self.collection,
                vector=query_vec,
                limit=top_k,
                with_payload=True,
            )
        else:
            raise RuntimeError(
                "Qdrant client does not have a compatible search/query method"
            )
        return [
            {
                "chunk_id": h.payload["chunk_id"],
                "doc_id": h.payload["doc_id"],
                "text": h.payload["text"],
                "source": h.payload.get("source", ""),
                "score": h.score,
                "chunk_index": h.payload.get("chunk_index", 0),
                "metadata": {
                    k: v
                    for k, v in h.payload.items()
                    if k not in {"chunk_id", "doc_id", "text", "source", "chunk_index"}
                },
            }
            for h in hits
        ]

    def collection_size(self) -> int:
        """Return the number of points currently in the collection."""
        info = self.client.get_collection(self.collection)
        return info.points_count or 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id_to_int(chunk_id: str) -> int:
    """Map a hex string ID to a stable positive integer for Qdrant point IDs."""
    return int(chunk_id, 16) % (2**63 - 1)
