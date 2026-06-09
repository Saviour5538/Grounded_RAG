"""End-to-end RAG pipeline orchestrator.

Phase 6: citation extraction + uncited-claim verification
Phase 7: in-memory LRU cache + JSONL request tracing (+ optional Langfuse)
Phase 8: agentic re-retrieval — on low confidence, reformulate query and retry
         up to max_retries times before final abstention

Full flow per query:
  cache check
  → hybrid retrieve (BM25 + dense + RRF, top-20)
  → Gemini reranker (keep top-5)
  → confidence gate:
      pass  → constrained LLM generation → citation verification
      fail  → [Phase 8] reformulate query → retry (up to N times) → or abstain
  → cache write + trace log
"""
from __future__ import annotations

import logging
import time
from typing import Any

from config.settings import Settings
from config.settings import settings as _default_settings
from grounded_rag.cache.query_cache import QueryCache
from grounded_rag.confidence.scorer import ConfidenceScorer
from grounded_rag.generation.citations import verify_citations
from grounded_rag.generation.generator import Generator
from grounded_rag.observability.tracer import Tracer
from grounded_rag.retrieval.dense import DenseRetriever, EmbeddingModel
from grounded_rag.retrieval.hybrid import HybridRetriever
from grounded_rag.retrieval.reformulator import QueryReformulator
from grounded_rag.retrieval.reranker import GeminiReranker
from grounded_rag.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)

_ABSTAIN = "Insufficient evidence found."


class RAGPipeline:
    """Thin orchestrator: cache → retrieve → rerank → confidence → generate → verify."""

    def __init__(self, settings: Settings | None = None):
        cfg = settings or _default_settings

        # ── Retrieval ──────────────────────────────────────────────────────────
        _embed_key = cfg.gemini_api_key if cfg.embedding_provider == "gemini" else cfg.openai_api_key
        embedder = EmbeddingModel(
            model_name=cfg.embedding_model,
            provider=cfg.embedding_provider,
            api_key=_embed_key,
        )

        if cfg.vector_store == "pinecone":
            from grounded_rag.retrieval.pinecone_retriever import PineconeRetriever
            dense = PineconeRetriever(
                api_key=cfg.pinecone_api_key,
                index_name=cfg.pinecone_index_name,
                embedding_model=embedder,
                dim=cfg.embedding_dim,
                cloud=cfg.pinecone_cloud,
                region=cfg.pinecone_region,
            )
            sparse = BM25Retriever(scroll_fn=dense.scroll_all)
        else:
            dense = DenseRetriever(
                collection=cfg.qdrant_collection,
                embedding_model=embedder,
                dim=cfg.embedding_dim,
                qdrant_url=cfg.qdrant_url,
                qdrant_mode=cfg.qdrant_mode,
                qdrant_local_path=cfg.qdrant_local_path,
                qdrant_api_key=cfg.qdrant_api_key,
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

        # ── Cache (Phase 7) ────────────────────────────────────────────────────
        self.cache = QueryCache(max_size=cfg.cache_max_size) if cfg.cache_enabled else None

        # ── Observability (Phase 7) ────────────────────────────────────────────
        self.tracer = Tracer(
            log_path=cfg.trace_log_path,
            langfuse_public_key=cfg.langfuse_public_key,
            langfuse_secret_key=cfg.langfuse_secret_key,
            langfuse_host=cfg.langfuse_host,
        )

        # ── Agentic re-retrieval (Phase 8) ────────────────────────────────────
        self.reformulator = (
            QueryReformulator(
                api_key=cfg.gemini_api_key,
                model=cfg.llm_model,
                max_retries=cfg.agentic_max_retries,
            )
            if cfg.agentic_retry_enabled else None
        )
        self._max_retries = cfg.agentic_max_retries if cfg.agentic_retry_enabled else 0

        self._gemini_api_key = cfg.gemini_api_key
        self._llm_model = cfg.llm_model
        self._retrieve_k = max(cfg.retrieval_top_k, cfg.reranker_top_n * 4)

    def query(self, question: str) -> dict[str, Any]:
        """Run the full pipeline: cache → retrieve → rerank → confidence → generate → verify."""
        t0 = time.perf_counter()

        # ── 0. Cache check ─────────────────────────────────────────────────────
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                cached["cache_hit"] = True
                cached["latency_ms"] = round((time.perf_counter() - t0) * 1000)
                self.tracer.log({
                    "query":      question,
                    "answer":     cached.get("answer", ""),
                    "latency_ms": cached["latency_ms"],
                    "cache_hit":  True,
                    "abstained":  cached.get("answer", "").startswith(_ABSTAIN),
                })
                return cached

        # ── 1-3. Retrieve → rerank → confidence (with agentic retry, Phase 8) ──
        active_query = question
        reformulations: list[str] = []
        chunks: list[dict[str, Any]] = []
        confidence: dict[str, Any] = {"score": 0.0, "signals": {}, "abstain": True}

        for attempt in range(self._max_retries + 1):
            candidates = self.retriever.retrieve(active_query, top_k=self._retrieve_k)

            if not candidates:
                break

            chunks = self.reranker.rerank(active_query, candidates)
            confidence = self.confidence_scorer.compute(active_query, chunks)

            if not confidence["abstain"]:
                break  # evidence is strong enough — proceed to generation

            # Confidence too low — try reformulating if retries remain
            if self.reformulator and attempt < self._max_retries:
                rewritten = self.reformulator.reformulate(question, attempt)
                if rewritten != active_query and rewritten not in reformulations:
                    reformulations.append(rewritten)
                    active_query = rewritten
                    logger.info(
                        "Retry %d/%d with reformulated query: %r",
                        attempt + 1, self._max_retries, rewritten[:80],
                    )
                else:
                    break  # same query — no point retrying

        if confidence["abstain"]:
            logger.info(
                "Abstaining after %d attempt(s) — final confidence %.3f",
                len(reformulations) + 1, confidence["score"],
            )
            return self._abstain(question, confidence, t0, reformulations)

        # ── 4. Generate ────────────────────────────────────────────────────────
        result = self.generator.generate(active_query, chunks)

        # ── 5. Citation verification (Phase 6) ────────────────────────────────
        citations = verify_citations(
            result["answer"], chunks,
            api_key=self._gemini_api_key,
            model=self._llm_model,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000)

        response: dict[str, Any] = {
            "answer":          result["answer"],
            "question":        question,
            "chunks":          chunks,
            "model":           result["model"],
            "usage":           result["usage"],
            "confidence":      confidence,
            "citations":       citations,
            "latency_ms":      latency_ms,
            "cache_hit":       False,
            "reformulations":  reformulations,
        }

        # ── 6. Cache write ─────────────────────────────────────────────────────
        if self.cache:
            self.cache.set(question, response)

        # ── 7. Trace ───────────────────────────────────────────────────────────
        self.tracer.log({
            "query":               question,
            "answer":              result["answer"],
            "latency_ms":          latency_ms,
            "n_candidates":        len(candidates),
            "n_reranked":          len(chunks),
            "confidence":          confidence["score"],
            "abstained":           False,
            "uncited_count":       citations["uncited_count"],
            "unsupported_count":   citations["unsupported_count"],
            "verification_passed": citations["verification_passed"],
            "reformulations":      reformulations,
            "cache_hit":           False,
        })

        return response

    def _abstain(
        self,
        question: str,
        confidence: dict,
        t0: float | None = None,
        reformulations: list[str] | None = None,
    ) -> dict[str, Any]:
        latency_ms = round((time.perf_counter() - t0) * 1000) if t0 is not None else 0
        response: dict[str, Any] = {
            "answer":         _ABSTAIN,
            "question":       question,
            "chunks":         [],
            "model":          None,
            "usage":          None,
            "confidence":     confidence,
            "citations":      None,
            "latency_ms":     latency_ms,
            "cache_hit":      False,
            "reformulations": reformulations or [],
        }
        if self.cache:
            self.cache.set(question, response)
        self.tracer.log({
            "query":          question,
            "answer":         _ABSTAIN,
            "latency_ms":     latency_ms,
            "abstained":      True,
            "confidence":     confidence.get("score", 0.0),
            "reformulations": reformulations or [],
            "cache_hit":      False,
        })
        return response
