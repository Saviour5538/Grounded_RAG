"""End-to-end RAG pipeline orchestrator.

Phase 4 + 5: hybrid retrieve → rerank → confidence gate → generate.
  Phase 3 → HybridRetriever (BM25 + dense + RRF)
  Phase 4 → GeminiReranker rescores top candidates, keeps top-n
  Phase 5 → ConfidenceScorer gates generation; abstains if score < threshold
  Phase 6 → citation verifier post-processes the answer (next)
"""
from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings
from config.settings import settings as _default_settings
from grounded_rag.confidence.scorer import ConfidenceScorer
from grounded_rag.generation.generator import Generator
from grounded_rag.retrieval.dense import DenseRetriever, EmbeddingModel
from grounded_rag.retrieval.hybrid import HybridRetriever
from grounded_rag.retrieval.reranker import GeminiReranker
from grounded_rag.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)

_ABSTAIN = "Insufficient evidence found."


class RAGPipeline:
    """Thin orchestrator: retrieve → rerank → confidence → generate."""

    def __init__(self, settings: Settings | None = None):
        cfg = settings or _default_settings

        # ── Retrieval ──────────────────────────────────────────────────────────
        _embed_key = cfg.gemini_api_key if cfg.embedding_provider == "gemini" else cfg.openai_api_key
        embedder = EmbeddingModel(
            model_name=cfg.embedding_model,
            provider=cfg.embedding_provider,
            api_key=_embed_key,
        )
        dense  = DenseRetriever(
            collection=cfg.qdrant_collection,
            embedding_model=embedder,
            dim=cfg.embedding_dim,
            qdrant_url=cfg.qdrant_url,
            qdrant_mode=cfg.qdrant_mode,
            qdrant_local_path=cfg.qdrant_local_path,
        )
        sparse = BM25Retriever(qdrant_client=dense.client, collection=cfg.qdrant_collection)
        self.retriever = HybridRetriever(dense=dense, sparse=sparse)

        # ── Reranker (Phase 4) ─────────────────────────────────────────────────
        self.reranker = GeminiReranker(
            api_key=cfg.gemini_api_key,
            model=cfg.llm_model,
            top_n=cfg.reranker_top_n,
        )

        # ── Confidence / Abstention (Phase 5) ─────────────────────────────────
        self.confidence_scorer = ConfidenceScorer(threshold=cfg.confidence_threshold)

        # ── Generation ─────────────────────────────────────────────────────────
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

        # Retrieve more candidates than we'll keep so the reranker has material
        self._retrieve_k = max(cfg.retrieval_top_k, cfg.reranker_top_n * 4)

    def query(self, question: str) -> dict[str, Any]:
        """Run the full pipeline: retrieve → rerank → confidence → generate."""
        logger.debug("Pipeline query: %r", question)

        # 1. Hybrid retrieval — fetch broad candidate set
        candidates = self.retriever.retrieve(question, top_k=self._retrieve_k)

        if not candidates:
            return self._abstain(question, {"score": 0.0, "signals": {}, "abstain": True})

        # 2. Rerank — cross-encoder scoring, keep top-n
        chunks = self.reranker.rerank(question, candidates)

        # 3. Confidence gate — abstain if evidence is too weak
        confidence = self.confidence_scorer.compute(question, chunks)
        if confidence["abstain"]:
            logger.info(
                "Abstaining — confidence %.3f < threshold %.3f",
                confidence["score"], self.confidence_scorer.threshold,
            )
            return self._abstain(question, confidence)

        # 4. Generate
        result = self.generator.generate(question, chunks)
        return {
            "answer":     result["answer"],
            "question":   question,
            "chunks":     chunks,
            "model":      result["model"],
            "usage":      result["usage"],
            "confidence": confidence,
        }

    def _abstain(self, question: str, confidence: dict) -> dict[str, Any]:
        return {
            "answer":     _ABSTAIN,
            "question":   question,
            "chunks":     [],
            "model":      None,
            "usage":      None,
            "confidence": confidence,
        }
