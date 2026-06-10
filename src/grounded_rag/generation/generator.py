"""LLM-based answer generation, strictly grounded in retrieved context.

The system prompt instructs the model to answer only from the provided passages
and to respond with the sentinel "Insufficient evidence found." when it cannot.
This constraint is enforced at the prompt level; Phase 5 adds a confidence gate
before the LLM is even called.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.

Rules:
1. Answer ONLY using the numbered context passages below.
2. Cite every factual claim with the passage number in [brackets], e.g. [1].
3. If the answer cannot be found in the passages, respond with exactly:
   "Insufficient evidence found."
4. Do not add information, opinions, or assumptions beyond the passages.
5. Be concise — one to three paragraphs maximum.\
"""


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        lines.append(f"[{i}] source={source!r}  score={score:.3f}\n{chunk['text']}")
    return "\n\n---\n\n".join(lines)


def _build_history_block(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in history:
        lines.append(f"Q: {turn['question']}")
        answer = turn["answer"]
        if len(answer) > 400:
            answer = answer[:400] + "…"
        lines.append(f"A: {answer}")
    return "\n".join(lines)


class Generator:
    """Thin wrapper over Anthropic, OpenAI, or Gemini chat completions.

    The same interface is used throughout the pipeline so the provider can be
    swapped via config without touching calling code.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        max_tokens: int = 1024,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")
        return self._client

    def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate an answer grounded in `chunks`.

        Returns a dict with keys: answer, model, usage, num_context_chunks.
        """
        context = _build_context_block(chunks)
        if history:
            history_block = _build_history_block(history)
            user_message = (
                f"Previous conversation:\n{history_block}\n\n"
                f"Context passages:\n\n{context}\n\n"
                f"Question: {query}"
            )
        else:
            user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"
        client = self._get_client()

        if self.provider == "anthropic":
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = resp.content[0].text
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        elif self.provider == "gemini":
            from google.genai import types as genai_types
            resp = client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                ),
            )
            answer = resp.text or ""
            meta = resp.usage_metadata
            usage = {
                "input_tokens": meta.prompt_token_count if meta else 0,
                "output_tokens": meta.candidates_token_count if meta else 0,
            }
        else:
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = resp.choices[0].message.content or ""
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }

        logger.debug("Generated answer (%d tokens out)", usage["output_tokens"])
        return {
            "answer": answer,
            "model": self.model,
            "usage": usage,
            "num_context_chunks": len(chunks),
        }

    def generate_stream(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> Iterator[str | dict[str, Any]]:
        """Stream the LLM answer token by token.

        Yields str pieces while generating, then yields one final dict
        {"answer", "model", "usage", "num_context_chunks"} as the last item.
        Callers distinguish str (token) from dict (sentinel) by type().
        """
        context = _build_context_block(chunks)
        if history:
            history_block = _build_history_block(history)
            user_message = (
                f"Previous conversation:\n{history_block}\n\n"
                f"Context passages:\n\n{context}\n\n"
                f"Question: {query}"
            )
        else:
            user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        client = self._get_client()

        if self.provider == "gemini":
            from google.genai import types as genai_types
            full_text: list[str] = []
            usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=user_message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                ),
            ):
                if chunk.text:
                    full_text.append(chunk.text)
                    yield chunk.text
                if getattr(chunk, "usage_metadata", None):
                    meta = chunk.usage_metadata
                    usage = {
                        "input_tokens":  meta.prompt_token_count or 0,
                        "output_tokens": meta.candidates_token_count or 0,
                    }
            yield {
                "answer":             "".join(full_text),
                "model":              self.model,
                "usage":              usage,
                "num_context_chunks": len(chunks),
            }

        elif self.provider == "anthropic":
            full_text = []
            with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    yield text
                resp = stream.get_final_message()
            yield {
                "answer":             "".join(full_text),
                "model":              self.model,
                "usage":              {
                    "input_tokens":  resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
                "num_context_chunks": len(chunks),
            }

        else:  # openai
            full_text = []
            stream = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content or ""
                if delta:
                    full_text.append(delta)
                    yield delta
            yield {
                "answer":             "".join(full_text),
                "model":              self.model,
                "usage":              {"input_tokens": 0, "output_tokens": 0},
                "num_context_chunks": len(chunks),
            }
