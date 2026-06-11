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

import json
import logging
import time
from typing import Any, Iterator

from config.settings import Settings
from config.settings import settings as _default_settings
from grounded_rag.cache.query_cache import QueryCache
from grounded_rag.confidence.scorer import ConfidenceScorer
from grounded_rag.generation.citations import (
    faithfulness_from_citations,
    score_answer_relevancy,
    verify_citations,
)
from grounded_rag.generation.generator import Generator
from grounded_rag.observability.tracer import Tracer
from grounded_rag.retrieval.dense import DenseRetriever, EmbeddingModel
from grounded_rag.retrieval.hybrid import HybridRetriever
from grounded_rag.retrieval.hyde import HyDEExpander
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

        # ── HyDE query expansion ──────────────────────────────────────────────
        self.hyde = (
            HyDEExpander(api_key=cfg.gemini_api_key, model=cfg.llm_model)
            if cfg.hyde_enabled else None
        )

        self._gemini_api_key = cfg.gemini_api_key
        self._llm_model = cfg.llm_model
        self._retrieve_k = max(cfg.retrieval_top_k, cfg.reranker_top_n * 4)

    def query(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline: cache → retrieve → rerank → confidence → generate → verify.

        history: list of {"question": str, "answer": str} from earlier turns in the
                 same session. Injected into the generation prompt so the LLM can
                 resolve references like "it" or "they" from prior context.
                 Cache is bypassed when history is present (session-specific answers
                 must not be served to other sessions).
        confidence_threshold: per-request override for the abstention threshold.
                 If None, uses the global setting from CONFIDENCE_THRESHOLD env var.
        """
        t0 = time.perf_counter()

        # ── 0. Cache check (skipped for session queries) ───────────────────────
        if self.cache and not history:
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
            dense_query = self.hyde.expand(active_query) if self.hyde else None
            candidates = self.retriever.retrieve(
                active_query, top_k=self._retrieve_k, dense_query=dense_query
            )

            if not candidates:
                break

            chunks = self.reranker.rerank(active_query, candidates)
            confidence = self.confidence_scorer.compute(
                active_query, chunks, threshold=confidence_threshold
            )

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
            return self._abstain(question, confidence, t0, reformulations, history=history)

        # ── 4. Generate ────────────────────────────────────────────────────────
        result = self.generator.generate(active_query, chunks, history=history)

        # ── 5. Citation verification (Phase 6) ────────────────────────────────
        citations = verify_citations(
            result["answer"], chunks,
            api_key=self._gemini_api_key,
            model=self._llm_model,
        )

        # ── 6. Per-query eval metrics ──────────────────────────────────────────
        faithfulness = faithfulness_from_citations(citations)
        answer_relevancy = score_answer_relevancy(
            question, result["answer"],
            api_key=self._gemini_api_key,
            model=self._llm_model,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000)

        response: dict[str, Any] = {
            "answer":           result["answer"],
            "question":         question,
            "chunks":           chunks,
            "model":            result["model"],
            "usage":            result["usage"],
            "confidence":       confidence,
            "citations":        citations,
            "faithfulness":     faithfulness,
            "answer_relevancy": answer_relevancy,
            "latency_ms":       latency_ms,
            "cache_hit":        False,
            "reformulations":   reformulations,
        }

        # ── 6. Cache write (skipped for session queries) ──────────────────────
        if self.cache and not history:
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

    def query_stream(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        confidence_threshold: float | None = None,
    ) -> Iterator[str]:
        """Synchronous generator that yields SSE-formatted strings.

        Event sequence:
          retrieval — confidence + chunks, sent immediately after reranking
          token     — one LLM output text piece
          meta      — full result (citations, metrics, latency, answer)
          done      — end of stream
          abstain   — emitted instead of token+meta when confidence too low
          error     — emitted on unexpected exception
        """

        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj, default=str)}\n\n"

        t0 = time.perf_counter()

        # ── 0. Cache check ────────────────────────────────────────────────────
        if self.cache and not history:
            cached = self.cache.get(question)
            if cached:
                latency_ms = round((time.perf_counter() - t0) * 1000)
                yield _sse({"type": "retrieval",
                             "confidence": cached.get("confidence", {}),
                             "chunks": cached.get("chunks", []),
                             "reformulations": []})
                yield _sse({"type": "token", "text": cached.get("answer", "")})
                yield _sse({"type": "meta",
                             "answer":           cached.get("answer"),
                             "model":            cached.get("model"),
                             "usage":            cached.get("usage"),
                             "confidence":       cached.get("confidence"),
                             "citations":        cached.get("citations"),
                             "faithfulness":     cached.get("faithfulness"),
                             "answer_relevancy": cached.get("answer_relevancy"),
                             "latency_ms":       latency_ms,
                             "chunks":           cached.get("chunks", []),
                             "cache_hit":        True,
                             "reformulations":   []})
                yield _sse({"type": "done"})
                return

        # ── 1-3. Retrieve → rerank → confidence (with agentic retry) ─────────
        active_query   = question
        reformulations: list[str]        = []
        candidates:     list[dict[str, Any]] = []
        chunks:         list[dict[str, Any]] = []
        confidence: dict[str, Any] = {"score": 0.0, "signals": {}, "abstain": True}

        try:
            for attempt in range(self._max_retries + 1):
                dense_query = self.hyde.expand(active_query) if self.hyde else None
                candidates = self.retriever.retrieve(
                    active_query, top_k=self._retrieve_k, dense_query=dense_query
                )
                if not candidates:
                    break
                chunks     = self.reranker.rerank(active_query, candidates)
                confidence = self.confidence_scorer.compute(
                    active_query, chunks, threshold=confidence_threshold
                )
                if not confidence["abstain"]:
                    break
                if self.reformulator and attempt < self._max_retries:
                    rewritten = self.reformulator.reformulate(question, attempt)
                    if rewritten != active_query and rewritten not in reformulations:
                        reformulations.append(rewritten)
                        active_query = rewritten
                    else:
                        break
        except Exception as exc:
            logger.error("Retrieval/rerank failed during stream: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})
            return

        yield _sse({"type": "retrieval",
                    "confidence":    confidence,
                    "chunks":        chunks,
                    "reformulations": reformulations})

        # ── Abstain path ──────────────────────────────────────────────────────
        if confidence["abstain"]:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            logger.info("Stream abstaining — confidence %.3f", confidence["score"])
            yield _sse({"type": "abstain",
                         "answer":         _ABSTAIN,
                         "confidence":     confidence,
                         "latency_ms":     latency_ms,
                         "reformulations": reformulations})
            yield _sse({"type": "done"})
            self.tracer.log({"query": question, "answer": _ABSTAIN, "abstained": True,
                              "confidence": confidence.get("score", 0.0),
                              "latency_ms": latency_ms, "reformulations": reformulations,
                              "cache_hit": False})
            return

        # ── 4. Stream LLM tokens ──────────────────────────────────────────────
        gen_result: dict[str, Any] = {}
        try:
            for piece in self.generator.generate_stream(active_query, chunks, history=history):
                if isinstance(piece, str):
                    yield _sse({"type": "token", "text": piece})
                else:
                    gen_result = piece  # final dict sentinel
        except Exception as exc:
            logger.error("Generation streaming failed: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})
            return

        answer = gen_result.get("answer", "")

        # ── 5. Citation verification + per-query metrics ──────────────────────
        citations      = verify_citations(answer, chunks,
                                          api_key=self._gemini_api_key,
                                          model=self._llm_model)
        faithfulness   = faithfulness_from_citations(citations)
        answer_relevancy = score_answer_relevancy(question, answer,
                                                   api_key=self._gemini_api_key,
                                                   model=self._llm_model)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        response: dict[str, Any] = {
            "answer":           answer,
            "question":         question,
            "chunks":           chunks,
            "model":            gen_result.get("model"),
            "usage":            gen_result.get("usage"),
            "confidence":       confidence,
            "citations":        citations,
            "faithfulness":     faithfulness,
            "answer_relevancy": answer_relevancy,
            "latency_ms":       latency_ms,
            "cache_hit":        False,
            "reformulations":   reformulations,
        }

        if self.cache and not history:
            self.cache.set(question, response)

        self.tracer.log({
            "query":               question,
            "answer":              answer,
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

        yield _sse({"type": "meta", **response})
        yield _sse({"type": "done"})

    def _abstain(
        self,
        question: str,
        confidence: dict,
        t0: float | None = None,
        reformulations: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
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
        if self.cache and not history:
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
