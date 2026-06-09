"""Download and prepare a corpus for indexing.

Default corpus: SciFact (BEIR) — ~5k scientific abstracts with fact-checking
claims, small enough to iterate on quickly, and it ships with Q/A pairs for
Phase 2 evals.

Usage:
    python scripts/ingest_corpus.py                          # SciFact (default)
    python scripts/ingest_corpus.py --corpus scifact --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_scifact(out_dir: Path) -> None:
    """Download the SciFact corpus and queries from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install the datasets library: pip install datasets")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Corpus ────────────────────────────────────────────────────────────────
    logger.info("Downloading SciFact corpus (~5 k documents)...")
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus", )
    corpus_path = out_dir / "scifact_corpus.jsonl"
    written = 0
    with corpus_path.open("w", encoding="utf-8") as f:
        for item in corpus_ds:
            title = (item.get("title") or "").strip()
            body = (item.get("text") or "").strip()
            text = f"{title}\n\n{body}".strip() if title else body
            if not text:
                continue
            record = {
                "id": str(item["_id"]),
                "text": text,
                "title": title,
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    logger.info("Wrote %d corpus documents → %s", written, corpus_path)

    # ── Queries (for Phase 2 eval) ─────────────────────────────────────────────
    logger.info("Downloading SciFact queries...")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    queries_path = out_dir / "scifact_queries.jsonl"
    written = 0
    with queries_path.open("w", encoding="utf-8") as f:
        for item in queries_ds:
            record = {"id": str(item["_id"]), "text": item["text"]}
            f.write(json.dumps(record) + "\n")
            written += 1
    logger.info("Wrote %d queries → %s", written, queries_path)

    # ── Qrels (claim → has evidence in corpus) ────────────────────────────────
    # Use ir_datasets which bundles BEIR qrels as tiny TSV files (no ML models).
    # Claims present in qrels have supporting/refuting evidence; absent = NOT_ENOUGH_INFO.
    logger.info("Downloading SciFact qrels via ir_datasets...")
    try:
        import ir_datasets

        claims_with_evidence: set[str] = set()
        for split_path in ("beir/scifact/train", "beir/scifact/test"):
            try:
                ds = ir_datasets.load(split_path)
                for qrel in ds.qrels_iter():
                    if qrel.relevance >= 1:
                        claims_with_evidence.add(str(qrel.query_id))
                logger.info("Loaded qrels from %s", split_path)
            except Exception as inner_e:
                logger.debug("Could not load %s: %s", split_path, inner_e)

        if claims_with_evidence:
            qrels_path = out_dir / "scifact_qrels.jsonl"
            with qrels_path.open("w", encoding="utf-8") as f:
                for claim_id in sorted(claims_with_evidence, key=lambda x: int(x) if x.isdigit() else x):
                    f.write(json.dumps({"claim_id": claim_id, "has_evidence": True}) + "\n")
            logger.info(
                "Wrote %d claims-with-evidence → %s", len(claims_with_evidence), qrels_path
            )
        else:
            logger.warning("No qrels found — abstention-correctness eval will be skipped")
    except ImportError:
        logger.warning("ir_datasets not installed — run: pip install ir-datasets")
    except Exception as e:
        logger.warning("Could not download qrels (%s) — abstention-correctness eval will be skipped", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a corpus for grounded-rag")
    parser.add_argument(
        "--corpus",
        default="scifact",
        choices=["scifact"],
        help="Which corpus to download (default: scifact)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    if args.corpus == "scifact":
        download_scifact(out)

    logger.info("Corpus ready at: %s", out.resolve())


if __name__ == "__main__":
    main()
