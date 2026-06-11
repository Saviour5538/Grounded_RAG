"""HyDE — Hypothetical Document Embeddings for query expansion.

Instead of embedding the raw query, generates a plausible hypothetical
answer via the LLM, then embeds that answer text. The embedding is
semantically closer to relevant documents than a short keyword query,
improving dense retrieval recall on complex questions.

Reference: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (HyDE).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PROMPT = (
    "Write a concise, factual paragraph that directly answers the question below. "
    "If you are unsure, write a plausible answer anyway — accuracy is secondary; "
    "semantic closeness to relevant documents is the goal.\n\n"
    "Question: {query}\n\nAnswer:"
)


class HyDEExpander:
    """Expands a query into a hypothetical answer for dense retrieval.

    Uses the same Gemini client as the rest of the pipeline — no extra API key needed.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._api_key = api_key
        self._model = model
        self._client: object = None

    def _load(self) -> None:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)

    def expand(self, query: str) -> str:
        """Return a hypothetical answer string to use as the dense retrieval query.

        Falls back to the original query on any API error so the pipeline
        degrades gracefully rather than failing.
        """
        self._load()
        try:
            from google.genai import types as _t
            response = self._client.models.generate_content(
                model=self._model,
                contents=_PROMPT.format(query=query),
                config=_t.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.3,
                ),
            )
            text = (response.text or "").strip()
            if text:
                logger.debug("HyDE expanded (%d chars): %s…", len(text), text[:80])
                return text
        except Exception as exc:
            logger.warning("HyDE expansion failed (%s) — using original query", exc)
        return query
