"""Shared utilities used across pipeline components."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    label: str = "Gemini",
) -> T:
    """Call fn() with exponential backoff on transient Gemini errors (503, 429).

    Raises the last exception if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            retryable = any(code in msg for code in _RETRYABLE)
            if not retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.0fs",
                label, attempt + 1, max_retries, msg[:100], delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # satisfies type checker
