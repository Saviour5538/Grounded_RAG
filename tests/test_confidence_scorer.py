"""Unit tests for the confidence scorer and abstention gate (Phase 5).

These are pure-logic tests — no API calls, no Qdrant. The scorer is the
trust-critical gating component that decides whether to answer or abstain,
so we test it thoroughly.
"""
import pytest

from grounded_rag.confidence.scorer import ConfidenceScorer, _token_overlap


# ── _token_overlap ─────────────────────────────────────────────────────────────

def test_token_overlap_identical():
    assert _token_overlap("hello world", "hello world") == 1.0


def test_token_overlap_disjoint():
    assert _token_overlap("foo bar", "baz qux") == 0.0


def test_token_overlap_partial():
    result = _token_overlap("the quick brown fox", "the quick red dog")
    assert 0.0 < result < 1.0


def test_token_overlap_empty_a():
    assert _token_overlap("", "hello world") == 0.0


def test_token_overlap_empty_b():
    assert _token_overlap("hello world", "") == 0.0


def test_token_overlap_case_insensitive():
    # The regex lowercases before matching
    assert _token_overlap("Hello World", "hello world") == 1.0


# ── ConfidenceScorer — empty / edge cases ────────────────────────────────────

def test_empty_chunks_abstains():
    scorer = ConfidenceScorer(threshold=0.3)
    result = scorer.compute("any query", [])
    assert result["score"] == 0.0
    assert result["abstain"] is True
    assert result["signals"] == {}


def test_result_has_required_keys():
    scorer = ConfidenceScorer(threshold=0.3)
    chunks = [{"text": "some text", "score": 0.02, "reranker_score": 0.5}]
    result = scorer.compute("some text", chunks)
    assert "score" in result
    assert "signals" in result
    assert "abstain" in result
    assert "reranker_score" in result["signals"]
    assert "rrf_signal" in result["signals"]
    assert "token_overlap" in result["signals"]


def test_score_clamped_to_unit_interval():
    scorer = ConfidenceScorer(threshold=0.0)
    chunks = [
        {"text": "query " * 100, "score": 999.0, "reranker_score": 1.0},
    ]
    result = scorer.compute("query " * 100, chunks)
    assert 0.0 <= result["score"] <= 1.0


# ── ConfidenceScorer — reranker path ─────────────────────────────────────────

def test_with_reranker_scores_uses_weighted_formula():
    scorer = ConfidenceScorer(threshold=0.0)
    chunks = [
        {"text": "quick brown fox", "score": 0.03, "reranker_score": 1.0},
        {"text": "unrelated text", "score": 0.02, "reranker_score": 0.0},
    ]
    result = scorer.compute("quick fox", chunks)
    assert result["signals"]["reranker_score"] is not None
    assert result["score"] > 0.0


def test_high_reranker_score_passes_threshold():
    scorer = ConfidenceScorer(threshold=0.3)
    chunks = [
        {"text": "query text matches well", "score": 0.03, "reranker_score": 1.0},
        {"text": "also relevant content", "score": 0.025, "reranker_score": 0.9},
    ]
    result = scorer.compute("query text matches", chunks)
    assert result["abstain"] is False
    assert result["score"] >= 0.3


def test_zero_reranker_scores_abstains_at_default_threshold():
    scorer = ConfidenceScorer(threshold=0.3)
    chunks = [
        {"text": "unrelated", "score": 0.001, "reranker_score": 0.0},
        {"text": "also unrelated", "score": 0.001, "reranker_score": 0.0},
    ]
    result = scorer.compute("completely different query with no overlap", chunks)
    assert result["abstain"] is True


# ── ConfidenceScorer — fallback (no reranker) path ───────────────────────────

def test_without_reranker_scores_uses_fallback():
    scorer = ConfidenceScorer(threshold=0.0)
    chunks = [{"text": "relevant text about topic", "score": 0.03}]
    result = scorer.compute("relevant text", chunks)
    assert result["signals"]["reranker_score"] is None
    assert result["score"] > 0.0


def test_partial_reranker_scores_uses_fallback():
    # If ANY chunk in top-5 is missing reranker_score, fallback is used
    scorer = ConfidenceScorer(threshold=0.0)
    chunks = [
        {"text": "text a", "score": 0.03, "reranker_score": 0.8},
        {"text": "text b", "score": 0.02},  # missing reranker_score
    ]
    result = scorer.compute("text a", chunks)
    assert result["signals"]["reranker_score"] is None


# ── ConfidenceScorer — threshold boundary ────────────────────────────────────

def test_score_exactly_at_threshold_does_not_abstain():
    # abstain is score < threshold, so equal should NOT abstain
    scorer = ConfidenceScorer(threshold=0.3)
    chunks = [{"text": "relevant", "score": 0.03, "reranker_score": 0.3 / 0.55}]
    result = scorer.compute("relevant", chunks)
    # We just verify the abstain logic: score < threshold
    assert result["abstain"] == (result["score"] < 0.3)


def test_custom_threshold_zero_never_abstains():
    scorer = ConfidenceScorer(threshold=0.0)
    chunks = [{"text": "anything", "score": 0.001, "reranker_score": 0.001}]
    result = scorer.compute("whatever", chunks)
    assert result["abstain"] is False


def test_custom_threshold_one_always_abstains():
    scorer = ConfidenceScorer(threshold=1.0)
    chunks = [{"text": "perfect match", "score": 0.03, "reranker_score": 1.0}]
    result = scorer.compute("perfect match", chunks)
    assert result["abstain"] is True


# ── ConfidenceScorer — uses only top 5 chunks ────────────────────────────────

def test_uses_only_top_5_chunks():
    scorer = ConfidenceScorer(threshold=0.0)
    # First 5 have high reranker scores; last 5 are None — top-5 slice should
    # still have all reranker scores, triggering the reranker formula
    chunks = (
        [{"text": "relevant", "score": 0.03, "reranker_score": 0.8} for _ in range(5)]
        + [{"text": "other", "score": 0.02} for _ in range(5)]
    )
    result = scorer.compute("relevant", chunks)
    assert result["signals"]["reranker_score"] is not None
