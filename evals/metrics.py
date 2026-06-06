"""Evaluation metrics for the RAG pipeline — Phase 2.

Metrics to implement:
  - retrieval_recall@k   : fraction of relevant chunks retrieved in top-k
  - context_precision    : RAGAS — are retrieved chunks actually relevant?
  - faithfulness         : RAGAS — does the answer stick to the context?
  - answer_relevancy     : RAGAS — does the answer address the question?
  - hallucination_rate   : custom — fraction of answer sentences not traceable
                           to any retrieved chunk
  - abstention_accuracy  : on a set of unanswerable questions, fraction where
                           the pipeline correctly returned "Insufficient evidence"

Each metric returns a float in [0, 1] (higher is better).
"""
from __future__ import annotations

# TODO Phase 2: implement retrieval and generation metrics
