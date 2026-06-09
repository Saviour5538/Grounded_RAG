"""Pinecone-backed dense retriever — drop-in replacement for DenseRetriever.

Uses Pinecone Serverless (free tier: 2 GB, forever). Stores chunk text and
metadata inside each vector's metadata dict so BM25Retriever can scroll all
chunks without a separate data store.

Setup:
  1. pip install pinecone
  2. Set VECTOR_STORE=pinecone, PINECONE_API_KEY, PINECONE_INDEX_NAME in .env
  3. Run scripts/build_index.py --clear to re-index into Pinecone
"""
from __future__ import annotations

import logging
import time
from typing import Any

from grounded_rag.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

_SAFE_METADATA_TYPES = (str, int, float, bool)


def _safe_metadata(meta: dict) -> dict:
    """Keep only Pinecone-compatible metadata values (str, int, float, bool, list[str])."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, _SAFE_METADATA_TYPES):
            out[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            out[k] = v
    return out


class PineconeRetriever:
    """Embed chunks with any EmbeddingModel, upsert to Pinecone, retrieve top-k.

    Has the same public interface as DenseRetriever so pipeline and scripts
    can swap between them with a single config flag.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: Any,
        dim: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.embedder = embedding_model
        self.dim = dim
        self.cloud = cloud
        self.region = region
        self._pc: Any = None
        self._index: Any = None

    def _load(self) -> None:
        if self._pc is None:
            from pinecone import Pinecone
            self._pc = Pinecone(api_key=self.api_key)

    def ensure_collection(self) -> None:
        """Create the Pinecone index if it doesn't already exist."""
        self._load()
        from pinecone import ServerlessSpec

        existing = [idx.name for idx in self._pc.list_indexes()]
        if self.index_name not in existing:
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            logger.info("Created Pinecone index '%s' (dim=%d) — waiting for ready...", self.index_name, self.dim)
            while not self._pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            logger.info("Pinecone index '%s' is ready", self.index_name)
        self._index = self._pc.Index(self.index_name)

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """Embed and upsert chunks into Pinecone. Returns count indexed."""
        if self._index is None:
            self.ensure_collection()

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch]).tolist()
            records = [
                {
                    "id": c.id,
                    "values": vec,
                    "metadata": _safe_metadata({
                        "chunk_id":    c.id,
                        "doc_id":      c.doc_id,
                        "text":        c.text,
                        "source":      c.source,
                        "chunk_index": c.chunk_index,
                        **c.metadata,
                    }),
                }
                for c, vec in zip(batch, vectors)
            ]
            self._index.upsert(vectors=records)
            total += len(records)
        return total

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return top-k chunks by cosine similarity to the query."""
        if self._index is None:
            self.ensure_collection()

        query_vec = self.embedder.embed([query], is_query=True)[0].tolist()
        result = self._index.query(vector=query_vec, top_k=top_k, include_metadata=True)

        return [
            {
                "chunk_id":    m.metadata["chunk_id"],
                "doc_id":      m.metadata["doc_id"],
                "text":        m.metadata["text"],
                "source":      m.metadata.get("source", ""),
                "score":       m.score,
                "chunk_index": m.metadata.get("chunk_index", 0),
                "metadata": {
                    k: v for k, v in m.metadata.items()
                    if k not in {"chunk_id", "doc_id", "text", "source", "chunk_index"}
                },
            }
            for m in result.matches
        ]

    def scroll_all(self) -> list[dict[str, Any]]:
        """Fetch every chunk from Pinecone — used by BM25Retriever to build its index."""
        if self._index is None:
            self.ensure_collection()

        logger.info("Scrolling all vectors from Pinecone index '%s' for BM25...", self.index_name)
        all_chunks: list[dict[str, Any]] = []

        for id_batch in self._index.list():
            fetch_result = self._index.fetch(ids=list(id_batch))
            for _vid, vec in fetch_result.vectors.items():
                m = vec.metadata
                all_chunks.append({
                    "chunk_id":    m["chunk_id"],
                    "doc_id":      m["doc_id"],
                    "text":        m["text"],
                    "source":      m.get("source", ""),
                    "score":       0.0,
                    "chunk_index": m.get("chunk_index", 0),
                    "metadata": {
                        k: v for k, v in m.items()
                        if k not in {"chunk_id", "doc_id", "text", "source", "chunk_index"}
                    },
                })

        logger.info("Scrolled %d chunks from Pinecone", len(all_chunks))
        return all_chunks

    def collection_size(self) -> int:
        if self._index is None:
            return 0
        stats = self._index.describe_index_stats()
        return stats.total_vector_count or 0

    def delete_index(self) -> None:
        """Drop the entire Pinecone index (used for --clear in build_index.py)."""
        self._load()
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self.index_name in existing:
            self._pc.delete_index(self.index_name)
            logger.info("Deleted Pinecone index '%s'", self.index_name)
        self._index = None
