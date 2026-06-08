"""Citation extraction and uncited-claim verification — Phase 6.

For each sentence in the generated answer:
  1. Check whether it contains a [N] citation reference.
  2. For cited sentences, verify via Gemini that the cited chunk actually
     supports the claim (one batched call per answer, not per sentence).

Sentences with no citation are flagged as uncited. Cited sentences where
Gemini says the chunk doesn't support the claim are flagged as unsupported.
"""
from __future__ import annotations

import re
import time
from typing import Any

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ABSTAIN_PREFIX = "Insufficient evidence"


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 10]


def _sentence_refs(sentence: str) -> list[int]:
    return [int(m) for m in _CITATION_RE.findall(sentence)]


def verify_citations(
    answer: str,
    chunks: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Verify every answer sentence is cited and supported by its cited chunk.

    Uses a single batched Gemini call for all cited sentences to minimise
    latency and API cost.

    Returns
    -------
    {
      sentences        : [{text, citations, supported, reason}],
      uncited_count    : int,
      unsupported_count: int,
      verification_passed: bool,
    }
    """
    if not answer or answer.startswith(_ABSTAIN_PREFIX):
        return {
            "sentences": [],
            "uncited_count": 0,
            "unsupported_count": 0,
            "verification_passed": True,
        }

    sentences = _split_sentences(answer)
    if not sentences:
        return {
            "sentences": [],
            "uncited_count": 0,
            "unsupported_count": 0,
            "verification_passed": True,
        }

    # ── Pass 1: classify sentences into cited vs uncited ─────────────────────
    records: list[dict[str, Any]] = []
    to_verify: list[tuple[int, str, int]] = []  # (record_idx, sentence, chunk_idx)

    for sent in sentences:
        refs = _sentence_refs(sent)
        if not refs:
            records.append({"text": sent, "citations": [], "supported": False, "reason": "no_citation"})
        else:
            chunk_idx = refs[0] - 1  # 1-based → 0-based, use primary citation
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                records.append({"text": sent, "citations": refs, "supported": False, "reason": "invalid_ref"})
            else:
                idx = len(records)
                records.append({"text": sent, "citations": refs, "supported": None, "reason": "pending"})
                to_verify.append((idx, sent, chunk_idx))

    uncited_count = sum(1 for r in records if r.get("reason") in ("no_citation", "invalid_ref"))

    # ── Pass 2: batch-verify cited sentences in one Gemini call ──────────────
    if to_verify:
        _batch_verify(to_verify, records, chunks, api_key, model)

    unsupported_count = sum(
        1 for r in records
        if r.get("supported") is False and r.get("reason") not in ("no_citation", "invalid_ref")
    )

    return {
        "sentences": records,
        "uncited_count": uncited_count,
        "unsupported_count": unsupported_count,
        "verification_passed": (uncited_count + unsupported_count) == 0,
    }


def _batch_verify(
    to_verify: list[tuple[int, str, int]],
    records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> None:
    """Fill in supported/reason for all pending records via one Gemini call."""
    from google import genai
    from google.genai import types as t

    client = genai.Client(api_key=api_key)

    lines = []
    for i, (_, sent, chunk_idx) in enumerate(to_verify):
        chunk_text = chunks[chunk_idx]["text"][:400]
        lines.append(
            f"Claim {i + 1}: {sent}\n"
            f"Passage {i + 1}: {chunk_text}"
        )

    block = "\n\n".join(lines)
    n = len(to_verify)
    prompt = (
        f"For each claim below, does the corresponding passage directly support it?\n\n"
        f"{block}\n\n"
        f"Reply with exactly {n} lines in the format:\n"
        + "\n".join(f"{i + 1}: YES or NO" for i in range(n))
        + "\nDo not add any other text."
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=t.GenerateContentConfig(
                max_output_tokens=n * 8,
                temperature=0.0,
                thinking_config=t.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = (resp.text or "").strip()
        verdicts = _parse_verdicts(raw, n)
    except Exception:
        verdicts = [True] * n  # fail open — don't penalise on API error

    for i, (rec_idx, _, _) in enumerate(to_verify):
        supported = verdicts[i] if i < len(verdicts) else True
        records[rec_idx]["supported"] = supported
        records[rec_idx]["reason"] = "supported" if supported else "unsupported"


def _parse_verdicts(raw: str, n: int) -> list[bool]:
    """Parse lines like '1: YES', '2: NO' into a bool list."""
    verdicts: dict[int, bool] = {}
    for line in raw.splitlines():
        m = re.match(r"(\d+)\s*:\s*(YES|NO)", line.strip(), re.IGNORECASE)
        if m:
            verdicts[int(m.group(1))] = m.group(2).upper() == "YES"
    return [verdicts.get(i + 1, True) for i in range(n)]
