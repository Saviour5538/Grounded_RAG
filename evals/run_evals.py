"""Evaluation harness runner — Phase 2.

Workflow:
  1. Load a Q/A dataset (e.g. SciFact queries + gold labels).
  2. For each question, run the full pipeline and collect answer + retrieved chunks.
  3. Score with metrics.py.
  4. Print a summary table; write results to evals/results/.

Run:
    python evals/run_evals.py --dataset scifact --out-dir evals/results/baseline
"""
from __future__ import annotations

# TODO Phase 2: implement eval runner
