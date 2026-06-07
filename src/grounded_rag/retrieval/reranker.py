"""Cross-encoder reranker using Gemini as the relevance judge — Phase 4.

Instead of a local cross-encoder model (bge-reranker, Qwen3-Reranker), we use
Gemini in a single batched prompt: send all candidate chunks + query, receive a
relevance score (0-10) for each chunk. Sort by score, return top-n.

This is the "LLM as reranker" (RankGPT-style) pattern used in production when
local cross-encoders are not available. One API call per query.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


class GeminiReranker:
    """Reranks retrieved chunks by relevance to the query using Gemini.

    Sends all candidates in one prompt, parses a JSON score array, and returns
    the top-n chunks with a normalised reranker_score (0-1) attached.
    """

    def __init__(self, api_key: str, model: str, top_n: int = 5):
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self._client: Any = None

    def _load(self) -> None:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score each chunk for relevance to query; return top_n sorted by score."""
        if not chunks:
            return chunks

        n = len(chunks)
        if n <= self.top_n:
            for c in chunks:
                c.setdefault("reranker_score", 0.5)
            return chunks

        self._load()

        passages = "\n\n".join(
            f"[{i}] {c['text'][:300]}" for i, c in enumerate(chunks)
        )
        prompt = (
            f"You are a relevance scorer for a scientific retrieval system.\n\n"
            f"Question: {query}\n\n"
            f"Rate each of the {n} passages below for how directly it helps answer "
            f"the question. Use integer scores: 0=irrelevant, 5=somewhat relevant, "
            f"10=directly answers the question.\n\n"
            f"{passages}\n\n"
            f"Reply with ONLY a JSON array of {n} integers, one per passage in order.\n"
            f"Do not include any other text."
        )

        try:
            from google.genai import types as t
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=t.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.0,
                    thinking_config=t.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw = (resp.text or "").strip()
            # Find all candidate arrays; pick the last one with correct length
            candidates = _SCORE_RE.findall(raw)
            m = None
            for candidate in reversed(candidates):
                arr = json.loads(candidate)
                if len(arr) == n:
                    m = arr
                    break
            if m is not None:
                scores = m
                if len(scores) == n:
                    scored = sorted(
                        zip(scores, chunks),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                    result = []
                    for score, chunk in scored[: self.top_n]:
                        c = dict(chunk)
                        c["reranker_score"] = round(float(score) / 10.0, 4)
                        result.append(c)
                    logger.debug(
                        "Reranker: %d → %d chunks, top score %.1f",
                        n, len(result), scores[0] if scores else 0,
                    )
                    return result
            logger.warning("Reranker: could not parse score array (n=%d) from: %r", n, raw[:200])
        except Exception as exc:
            logger.warning("Reranker failed (%s) — falling back to score order", exc)

        # Fallback: score order, mark as unscored
        result = []
        for chunk in chunks[: self.top_n]:
            c = dict(chunk)
            c["reranker_score"] = c.get("score", 0.0)
            result.append(c)
        return result
