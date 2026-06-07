"""Evaluation metrics for the RAG pipeline — Phase 2.

All scoring is done via the Gemini API — no RAGAS, no local models.
Implementing the metrics ourselves demonstrates understanding of the internals
(see CLAUDE.md: "avoid heavy abstractions in retrieval, reranking, generation").

Metrics
-------
faithfulness          : fraction of answer sentences supported by retrieved context
answer_relevancy      : how well the answer addresses the question (0–1)
context_precision     : fraction of retrieved chunks relevant to the question
hallucination_rate    : fraction of answers with at least one unsupported claim
abstention_rate       : fraction of queries returning "Insufficient evidence"
avg_retrieval_score   : mean cosine similarity of top-retrieved chunks
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import settings

logger = logging.getLogger(__name__)

_ABSTAIN = "Insufficient evidence"
_DELAY   = 0.3   # seconds between Gemini calls to stay within rate limits


def _gemini_client():
    from google import genai
    return genai.Client(api_key=settings.gemini_api_key)


def _ask(client, prompt: str) -> str:
    """Single-turn Gemini call. Returns stripped text."""
    from google.genai import types as t
    resp = client.models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=t.GenerateContentConfig(max_output_tokens=16, temperature=0.0),
    )
    return (resp.text or "").strip()


# ── Individual metric scorers ─────────────────────────────────────────────────

def faithfulness_score(sample: dict, client) -> float:
    """Fraction of answer sentences that are supported by the retrieved context.

    Method (RAGAS-equivalent):
      1. Split the answer into sentences.
      2. For each sentence, ask Gemini: "Is this sentence supported by the context? YES/NO"
      3. Score = supported / total.
    """
    answer   = sample["answer"]
    contexts = sample["contexts"]

    if not answer or not contexts:
        return 0.0

    # Sentence-split by period/exclamation/question (simple heuristic is fine)
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    context_block = "\n\n".join(contexts[:5])
    supported = 0

    for sent in sentences:
        prompt = (
            "Given the following context, is the statement below directly supported "
            "by information in the context?\n\n"
            f"Context:\n{context_block}\n\n"
            f"Statement: {sent}\n\n"
            "Reply with only YES or NO."
        )
        verdict = _ask(client, prompt).upper()
        if verdict.startswith("YES"):
            supported += 1
        time.sleep(_DELAY)

    return round(supported / len(sentences), 4)


def answer_relevancy_score(sample: dict, client) -> float:
    """How well does the answer address the question? (0 = not at all, 1 = perfectly)

    Method:
      Ask Gemini to rate relevance 0–10, then normalise.
    """
    prompt = (
        "Rate how well the answer addresses the question on a scale of 0 to 10.\n"
        "0 = completely off-topic, 10 = perfectly and completely answers the question.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Answer: {sample['answer']}\n\n"
        "Reply with a single integer between 0 and 10."
    )
    raw = _ask(client, prompt)
    time.sleep(_DELAY)
    import re
    m = re.search(r"\d+", raw)
    if not m:
        return 0.5
    return round(min(int(m.group()), 10) / 10, 4)


def context_precision_score(sample: dict, client) -> float:
    """Fraction of retrieved chunks that are relevant to the question.

    Method: ask Gemini whether each chunk helps answer the question (YES/NO).
    Weighted by rank: top chunks count more (standard RAGAS definition).
    """
    question = sample["question"]
    contexts = sample["contexts"]

    if not contexts:
        return 0.0

    relevant_flags = []
    for ctx in contexts[:5]:
        prompt = (
            f"Does the following text contain information useful for answering "
            f"the question below?\n\n"
            f"Question: {question}\n\n"
            f"Text: {ctx}\n\n"
            "Reply with only YES or NO."
        )
        verdict = _ask(client, prompt).upper()
        relevant_flags.append(verdict.startswith("YES"))
        time.sleep(_DELAY)

    # Weighted precision@k: sum of (precision up to k * relevance_k) / relevant_total
    n = len(relevant_flags)
    score_sum = 0.0
    running_relevant = 0
    for i, is_rel in enumerate(relevant_flags):
        if is_rel:
            running_relevant += 1
            score_sum += running_relevant / (i + 1)

    if running_relevant == 0:
        return 0.0
    return round(score_sum / running_relevant, 4)


def is_hallucinated(sample: dict, client) -> bool:
    """Does the answer contain claims not supported by the retrieved context?"""
    context_block = "\n\n".join(sample["contexts"][:5]) if sample["contexts"] else "(none)"
    prompt = (
        "You are a hallucination detector for a retrieval-augmented system.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Generated answer:\n{sample['answer']}\n\n"
        "Does the answer contain any specific fact, claim, or number NOT present "
        "in the retrieved context? Reply with only YES or NO."
    )
    verdict = _ask(client, prompt).upper()
    time.sleep(_DELAY)
    return verdict.startswith("YES")


# ── Main scorers ──────────────────────────────────────────────────────────────

def run_ragas(samples: list[dict]) -> dict[str, float]:
    """Compute faithfulness, answer_relevancy, and context_precision.

    Skips abstained answers. Returns metric_name → mean float in [0, 1].
    """
    answerable = [s for s in samples if not s["answer"].startswith(_ABSTAIN)]
    if not answerable:
        logger.warning("All samples abstained — no generation metrics to compute")
        return {}

    client = _gemini_client()
    faithfulness_scores     = []
    answer_relevancy_scores = []
    context_precision_scores = []

    n = len(answerable)
    for i, s in enumerate(answerable):
        logger.info("[RAGAS %d/%d] %s…", i + 1, n, s["question"][:60])
        faithfulness_scores.append(faithfulness_score(s, client))
        answer_relevancy_scores.append(answer_relevancy_score(s, client))
        context_precision_scores.append(context_precision_score(s, client))

    def mean(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "faithfulness":      mean(faithfulness_scores),
        "answer_relevancy":  mean(answer_relevancy_scores),
        "context_precision": mean(context_precision_scores),
    }


def run_custom_metrics(samples: list[dict]) -> dict[str, float]:
    """Compute abstention_rate, hallucination_rate, and avg_retrieval_score."""
    n = len(samples)
    if n == 0:
        return {}

    abstained = [s for s in samples if s["answer"].startswith(_ABSTAIN)]
    answered  = [s for s in samples if not s["answer"].startswith(_ABSTAIN)]
    abstention_rate = len(abstained) / n

    avg_score = (
        sum(s.get("mean_chunk_score", 0.0) for s in samples) / n
    )

    # Hallucination check on answered samples only
    if answered:
        client = _gemini_client()
        hallucinated = sum(1 for s in answered if is_hallucinated(s, client))
        hallucination_rate = hallucinated / len(answered)
    else:
        hallucination_rate = 0.0

    return {
        "abstention_rate":    round(abstention_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_retrieval_score": round(avg_score, 4),
        "n_answered":  len(answered),
        "n_abstained": len(abstained),
    }
