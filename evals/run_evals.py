"""Evaluation harness runner — Phase 2.

Runs the RAG pipeline over a sample of SciFact claims, scores with RAGAS
(faithfulness, answer relevancy, context precision) plus custom metrics
(hallucination rate, abstention rate), and saves a baseline JSON report.

All scoring uses the Gemini API — no local models.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --n-samples 50 --out evals/results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from evals.datasets.scifact_qa import load_eval_samples
from evals.metrics import run_custom_metrics, run_ragas

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_pipeline_outputs(
    queries: list[dict],
    api_url: str = "http://localhost:8000",
    inter_query_delay: float = 0.5,
) -> list[dict]:
    """Call the running API /query endpoint for each query.

    This avoids the Qdrant file-lock conflict that occurs when two processes
    try to open the same local storage simultaneously.
    """
    import httpx

    results = []
    n = len(queries)

    with httpx.Client(base_url=api_url, timeout=60.0) as client:
        for i, q in enumerate(queries):
            logger.info("[%d/%d] %s", i + 1, n, q["question"][:80])
            try:
                resp = client.post("/query", json={"question": q["question"]})
                resp.raise_for_status()
                out = resp.json()
                chunks = out.get("chunks", [])
                mean_score = (
                    sum(c.get("score", 0.0) for c in chunks) / len(chunks)
                    if chunks else 0.0
                )
                results.append({
                    "question":         q["question"],
                    "claim_id":         q.get("claim_id", ""),
                    "answer":           out["answer"],
                    "contexts":         [c["text"] for c in chunks],
                    "chunk_sources":    [c.get("source", "") for c in chunks],
                    "mean_chunk_score": round(mean_score, 4),
                })
            except Exception as exc:
                logger.warning("API error on sample %d: %s", i, exc)
                results.append({
                    "question":         q["question"],
                    "claim_id":         q.get("claim_id", ""),
                    "answer":           "ERROR: " + str(exc),
                    "contexts":         [],
                    "chunk_sources":    [],
                    "mean_chunk_score": 0.0,
                })

            if i < n - 1:
                time.sleep(inter_query_delay)

    return results


def print_summary(ragas_scores: dict, custom_scores: dict) -> None:
    print("\n" + "=" * 52)
    print("  PHASE 2 BASELINE — GROUNDED RAG")
    print("=" * 52)
    print("\n  RAGAS Metrics  (higher = better)")
    print("  " + "-" * 40)
    for k, v in ragas_scores.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<36} {v:.3f}  {bar}")

    print("\n  Custom Metrics")
    print("  " + "-" * 40)
    for k, v in custom_scores.items():
        if isinstance(v, float):
            print(f"  {k:<36} {v:.3f}")
        else:
            print(f"  {k:<36} {v}")
    print("=" * 52 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 eval harness")
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of SciFact claims to evaluate (default: 50)"
    )
    parser.add_argument(
        "--out", default="evals/results/baseline.json",
        help="Output path for the JSON results (default: evals/results/baseline.json)"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="Base URL of the running RAG API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS scoring (runs custom metrics only, faster)"
    )
    args = parser.parse_args()

    # ── 1. Load eval queries ──────────────────────────────────────────────────
    logger.info("Loading %d eval samples from SciFact…", args.n_samples)
    queries = load_eval_samples(n=args.n_samples)
    logger.info("Loaded %d queries", len(queries))

    # ── 2. Run pipeline via API (avoids Qdrant file-lock conflict) ────────────
    logger.info("Calling API at %s for %d queries…", args.api_url, len(queries))
    samples = collect_pipeline_outputs(queries, api_url=args.api_url)
    answered  = sum(1 for s in samples if not s["answer"].startswith("Insufficient"))
    abstained = len(samples) - answered
    logger.info("Pipeline done — answered: %d | abstained: %d", answered, abstained)

    # ── 4. Score ──────────────────────────────────────────────────────────────
    ragas_scores: dict = {}
    if not args.skip_ragas:
        logger.info("Running RAGAS metrics (Gemini as judge)…")
        ragas_scores = run_ragas(samples)

    logger.info("Running custom metrics…")
    custom_scores = run_custom_metrics(samples)

    # ── 5. Save results ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "phase": "2_baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "config": {
            "embedding_model": settings.embedding_model,
            "llm_model":       settings.llm_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "chunk_size":      settings.chunk_size,
        },
        "ragas":  ragas_scores,
        "custom": custom_scores,
        "samples": samples,   # full trace for debugging
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print_summary(ragas_scores, custom_scores)


if __name__ == "__main__":
    main()
