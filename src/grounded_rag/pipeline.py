"""End-to-end RAG pipeline orchestrator.

Phase 3: hybrid retrieve (BM25 + dense, RRF-fused) → generate.
Later phases slot in without changing the public interface:
  Phase 4 → reranker.py added between retrieve and generate
  Phase 5 → confidence scorer gates generation / triggers abstention
  Phase 6 → citation verifier post-processes the answer
"""
from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings
from config.settings import settings as _default_settings
from grounded_rag.generation.generator import Generator
from grounded_rag.retrieval.dense import DenseRetriever, EmbeddingModel
from grounded_rag.retrieval.hybrid import HybridRetriever
from grounded_rag.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)

_ABSTAIN = "Insufficient evidence found."


class RAGPipeline:
    """Thin orchestrator wiring together retrieval and generation."""

    def __init__(self, settings: Settings | None = None):
        cfg = settings or _default_settings
        _embed_key = cfg.gemini_api_key if cfg.embedding_provider == "gemini" else cfg.openai_api_key
        embedder = EmbeddingModel(
            model_name=cfg.embedding_model,
            provider=cfg.embedding_provider,
            api_key=_embed_key,
        )
        dense = DenseRetriever(
            collection=cfg.qdrant_collection,
            embedding_model=embedder,
            dim=cfg.embedding_dim,
            qdrant_url=cfg.qdrant_url,
            qdrant_mode=cfg.qdrant_mode,
            qdrant_local_path=cfg.qdrant_local_path,
        )
        sparse = BM25Retriever(qdrant_client=dense.client, collection=cfg.qdrant_collection)
        self.retriever = HybridRetriever(dense=dense, sparse=sparse)

        _api_key_map = {
            "anthropic": cfg.anthropic_api_key,
            "openai":    cfg.openai_api_key,
            "gemini":    cfg.gemini_api_key,
        }
        self.generator = Generator(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            api_key=_api_key_map.get(cfg.llm_provider, ""),
            max_tokens=cfg.llm_max_tokens,
        )
        self.top_k = cfg.retrieval_top_k

    def query(self, question: str) -> dict[str, Any]:
        """Run the full pipeline for one question.

        Returns a dict with keys: answer, question, chunks, model, usage.
        """
        logger.debug("Pipeline query: %r", question)
        chunks = self.retriever.retrieve(question, top_k=self.top_k)

        if not chunks:
            logger.info("No chunks retrieved — returning abstention")
            return {
                "answer":   _ABSTAIN,
                "question": question,
                "chunks":   [],
                "model":    None,
                "usage":    None,
            }

        result = self.generator.generate(question, chunks)
        return {
            "answer":   result["answer"],
            "question": question,
            "chunks":   chunks,
            "model":    result["model"],
            "usage":    result["usage"],
        }
