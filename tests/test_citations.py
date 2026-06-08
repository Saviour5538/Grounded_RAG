"""Unit tests for the citation extraction and verification module (Phase 6).

The citation verifier is trust-critical: it flags uncited claims and calls
Gemini to confirm cited passages actually support the claim. We test:
  - Pure logic helpers (_split_sentences, _sentence_refs, _parse_verdicts)
  - verify_citations short-circuits on abstain answers
  - Uncited / invalid-ref counting (no Gemini call needed)
  - Gemini-call path with a mocked client
  - Fail-open behaviour on API errors
"""
from unittest.mock import MagicMock, patch

import pytest

from grounded_rag.generation.citations import (
    _parse_verdicts,
    _sentence_refs,
    _split_sentences,
    verify_citations,
)


# ── _split_sentences ──────────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "First sentence here. Second sentence here. Third one too."
    parts = _split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_filters_short_fragments():
    sentences = _split_sentences("OK. This is a real sentence here.")
    assert all(len(s) > 10 for s in sentences)


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_question_and_exclamation():
    text = "Is this correct? Yes it is! And that matters here."
    parts = _split_sentences(text)
    assert len(parts) == 3


# ── _sentence_refs ────────────────────────────────────────────────────────────

def test_sentence_refs_single():
    assert _sentence_refs("Claim is supported [1].") == [1]


def test_sentence_refs_multiple():
    assert _sentence_refs("Supported by [1] and [3].") == [1, 3]


def test_sentence_refs_none():
    assert _sentence_refs("No citation anywhere here.") == []


def test_sentence_refs_two_digit():
    assert _sentence_refs("As shown in [12] and [4].") == [12, 4]


# ── _parse_verdicts ───────────────────────────────────────────────────────────

def test_parse_verdicts_all_yes():
    assert _parse_verdicts("1: YES\n2: YES\n3: YES", 3) == [True, True, True]


def test_parse_verdicts_all_no():
    assert _parse_verdicts("1: NO\n2: NO", 2) == [False, False]


def test_parse_verdicts_mixed():
    assert _parse_verdicts("1: YES\n2: NO\n3: YES", 3) == [True, False, True]


def test_parse_verdicts_case_insensitive():
    assert _parse_verdicts("1: yes\n2: no", 2) == [True, False]


def test_parse_verdicts_missing_entry_defaults_true():
    # Missing index → fail open (True)
    verdicts = _parse_verdicts("1: YES", 3)
    assert verdicts == [True, True, True]


def test_parse_verdicts_ignores_extra_text():
    raw = "Here are the results:\n1: YES\n2: NO\nEnd."
    assert _parse_verdicts(raw, 2) == [True, False]


# ── verify_citations — no-API-call paths ─────────────────────────────────────

def test_abstain_answer_returns_empty_passed():
    result = verify_citations(
        "Insufficient evidence found.",
        chunks=[],
        api_key="fake",
        model="fake",
    )
    assert result["sentences"] == []
    assert result["uncited_count"] == 0
    assert result["unsupported_count"] == 0
    assert result["verification_passed"] is True


def test_empty_answer_returns_empty_passed():
    result = verify_citations("", chunks=[], api_key="fake", model="fake")
    assert result["sentences"] == []
    assert result["verification_passed"] is True


def test_uncited_sentence_counted():
    answer = "This sentence has absolutely no citation at all and is long enough."
    result = verify_citations(answer, chunks=[], api_key="fake", model="fake")
    assert result["uncited_count"] == 1
    assert result["verification_passed"] is False


def test_invalid_ref_out_of_range_counted_as_uncited():
    answer = "This references chunk [99] which does not exist in the result set."
    chunks = [{"text": "only one chunk available here for testing purposes"}]
    result = verify_citations(answer, chunks=chunks, api_key="fake", model="fake")
    assert result["uncited_count"] == 1


def test_multiple_uncited_sentences():
    answer = (
        "First uncited claim about something important. "
        "Second uncited claim about something else entirely."
    )
    result = verify_citations(answer, chunks=[], api_key="fake", model="fake")
    assert result["uncited_count"] == 2


# ── verify_citations — Gemini-call path (mocked) ────────────────────────────

def test_cited_supported_sentence_passes():
    answer = "This claim is directly supported by the evidence [1]."
    chunks = [{"text": "Evidence that directly supports the claim mentioned here."}]

    mock_resp = MagicMock()
    mock_resp.text = "1: YES"

    with patch("google.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_cls.return_value = mock_client

        result = verify_citations(answer, chunks=chunks, api_key="fake", model="m")

    assert result["uncited_count"] == 0
    assert result["unsupported_count"] == 0
    assert result["verification_passed"] is True


def test_cited_but_unsupported_sentence_flagged():
    answer = "This false claim is cited as [1] but the chunk disagrees."
    chunks = [{"text": "Unrelated content that does not support the claim at all."}]

    mock_resp = MagicMock()
    mock_resp.text = "1: NO"

    with patch("google.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_cls.return_value = mock_client

        result = verify_citations(answer, chunks=chunks, api_key="fake", model="m")

    assert result["unsupported_count"] == 1
    assert result["verification_passed"] is False


def test_api_error_fails_open():
    answer = "This cited claim [1] should not fail hard on an API error."
    chunks = [{"text": "Supporting text that would back up the cited claim here."}]

    with patch("google.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("network error")
        mock_cls.return_value = mock_client

        result = verify_citations(answer, chunks=chunks, api_key="fake", model="m")

    # Fail open: API error must NOT mark as unsupported
    assert result["unsupported_count"] == 0
    assert result["verification_passed"] is True


def test_mixed_cited_and_uncited():
    answer = (
        "This cited claim references [1]. "
        "This second sentence has no citation at all anywhere."
    )
    chunks = [{"text": "Chunk text supporting the first claim adequately."}]

    mock_resp = MagicMock()
    mock_resp.text = "1: YES"

    with patch("google.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_cls.return_value = mock_client

        result = verify_citations(answer, chunks=chunks, api_key="fake", model="m")

    assert result["uncited_count"] == 1
    assert result["unsupported_count"] == 0
    assert result["verification_passed"] is False
