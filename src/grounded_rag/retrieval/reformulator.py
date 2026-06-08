"""Agentic query reformulation — Phase 8.

When the confidence gate would abstain, this module asks Gemini to rewrite
the query in a way that might retrieve better evidence. The pipeline retries
up to `max_retries` times before giving up.

Each attempt uses a different reformulation strategy so retries are not just
paraphrases of each other:
  attempt 1 → expand: add synonyms and context terms
  attempt 2 → decompose: break a compound question into its core sub-claim
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_STRATEGIES = [
    (
        "expand",
        "Rewrite the following question to be more specific and detailed, "
        "adding relevant scientific terminology and context that would help "
        "retrieve supporting evidence from a biomedical corpus. "
        "Return ONLY the rewritten question, no explanation.",
    ),
    (
        "decompose",
        "Simplify the following question to its single most important factual "
        "sub-claim that can be verified against scientific literature. "
        "Return ONLY the simplified question, no explanation.",
    ),
]


class QueryReformulator:
    """Uses Gemini to rewrite a low-confidence query before retrying retrieval.

    Parameters
    ----------
    api_key    : Gemini API key
    model      : Gemini model name
    max_retries: number of reformulation attempts before final abstention
    """

    def __init__(self, api_key: str, model: str, max_retries: int = 2):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self._client: Any = None

    def _load(self) -> None:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def reformulate(self, query: str, attempt: int) -> str:
        """Return a rewritten version of `query` for the given attempt index.

        Falls back to the original query if Gemini fails.
        """
        if attempt >= len(_STRATEGIES):
            return query

        strategy_name, instruction = _STRATEGIES[attempt]
        self._load()

        prompt = f"{instruction}\n\nOriginal question: {query}"
        try:
            from google.genai import types as t
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=t.GenerateContentConfig(
                    max_output_tokens=128,
                    temperature=0.2,
                    thinking_config=t.ThinkingConfig(thinking_budget=0),
                ),
            )
            rewritten = (resp.text or "").strip().strip('"').strip("'")
            if rewritten and rewritten != query:
                logger.info(
                    "Reformulated query [%s]: %r → %r",
                    strategy_name, query[:60], rewritten[:60],
                )
                return rewritten
        except Exception as exc:
            logger.warning("Reformulation failed (attempt %d): %s", attempt, exc)

        return query
