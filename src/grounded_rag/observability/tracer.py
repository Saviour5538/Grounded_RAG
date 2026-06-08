"""Request tracing — Phase 7.

Writes one JSON line per request to a local JSONL file. Each trace captures
the full retrieval path so any failure is reproducible from logs alone:

  query → n_candidates → reranker_scores → confidence → citation check
        → answer length → latency_ms → cache_hit

Optionally forwards to Langfuse for real-time dashboards if credentials
are configured in settings (langfuse_public_key / langfuse_secret_key).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracer:
    """Append-only JSONL trace logger with optional Langfuse forwarding.

    Parameters
    ----------
    log_path            : path to the JSONL file (parent dirs created if needed)
    langfuse_public_key : if set, traces are also sent to Langfuse
    langfuse_secret_key : Langfuse secret
    langfuse_host       : Langfuse host URL
    """

    def __init__(
        self,
        log_path: str = "logs/traces.jsonl",
        langfuse_public_key: str = "",
        langfuse_secret_key: str = "",
        langfuse_host: str = "https://cloud.langfuse.com",
    ):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lf = None

        if langfuse_public_key:
            try:
                from langfuse import Langfuse
                self._lf = Langfuse(
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_host,
                )
                logger.info("Langfuse tracing enabled → %s", langfuse_host)
            except ImportError:
                logger.warning(
                    "langfuse not installed — install it with: pip install langfuse"
                )

    def log(self, trace: dict[str, Any]) -> None:
        """Append one trace record to the JSONL file and optionally Langfuse."""
        record = {"ts": datetime.now(timezone.utc).isoformat(), **trace}

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        if self._lf:
            try:
                self._lf.generation(
                    name="rag_query",
                    input=trace.get("query"),
                    output=trace.get("answer"),
                    metadata={k: v for k, v in trace.items() if k not in {"query", "answer"}},
                )
            except Exception as exc:
                logger.warning("Langfuse trace failed: %s", exc)
