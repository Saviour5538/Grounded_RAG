"""Build the Qdrant vector index from processed corpus data.

Loads all .jsonl / .csv / .txt files from a directory, deduplicates, chunks,
embeds via the Gemini/OpenAI API, and upserts into Qdrant.
Run after `scripts/ingest_corpus.py`.

Usage:
    python scripts/ingest_corpus.py             # download corpus
    python scripts/build_index.py               # embed + index
    python scripts/build_index.py --clear       # wipe existing index first
    python scripts/build_index.py --data-dir data/processed --batch-size 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Make project root importable.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from grounded_rag.ingest.chunker import chunk_document
from grounded_rag.ingest.dedup import dedup_documents
from grounded_rag.ingest.loader import load_directory
from grounded_rag.ingest.metadata import enrich_document
from grounded_rag.retrieval.dense import EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_EXCLUDE_PATTERNS = {"*_queries.jsonl", "*_eval.jsonl"}


def build_index(data_dir: Path, batch_size: int = 50, clear: bool = False) -> None:
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.error("Run `python scripts/ingest_corpus.py` first.")
        sys.exit(1)

    logger.info("Loading documents from %s", data_dir)
    raw_docs = [
        d for d in load_directory(data_dir)
        if not any(Path(d.source).match(pat) for pat in _EXCLUDE_PATTERNS)
    ]
    logger.info("Loaded %d documents (excluded eval/query files)", len(raw_docs))
    enriched = (enrich_document(d) for d in dedup_documents(raw_docs))

    _embed_key = settings.gemini_api_key if settings.embedding_provider == "gemini" else settings.openai_api_key
    embedder = EmbeddingModel(
        model_name=settings.embedding_model,
        provider=settings.embedding_provider,
        api_key=_embed_key,
    )

    if settings.vector_store == "pinecone":
        from grounded_rag.retrieval.pinecone_retriever import PineconeRetriever
        retriever = PineconeRetriever(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            embedding_model=embedder,
            dim=settings.embedding_dim,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        if clear:
            retriever.delete_index()
    else:
        from grounded_rag.retrieval.dense import DenseRetriever
        retriever = DenseRetriever(
            collection=settings.qdrant_collection,
            embedding_model=embedder,
            dim=settings.embedding_dim,
            qdrant_url=settings.qdrant_url,
            qdrant_mode=settings.qdrant_mode,
            qdrant_local_path=settings.qdrant_local_path,
            qdrant_api_key=settings.qdrant_api_key,
        )
        if clear:
            existing = {c.name for c in retriever.client.get_collections().collections}
            if settings.qdrant_collection in existing:
                retriever.client.delete_collection(settings.qdrant_collection)
                logger.info("Cleared existing collection '%s'", settings.qdrant_collection)

    retriever.ensure_collection()

    pending: list = []
    doc_count = 0
    chunk_count = 0

    for doc in tqdm(enriched, desc="Chunking", unit="doc"):
        chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        pending.extend(chunks)
        doc_count += 1

        if len(pending) >= batch_size:
            indexed = retriever.index_chunks(pending)
            chunk_count += indexed
            logger.info("Indexed %d chunks (docs so far: %d)", chunk_count, doc_count)
            pending = []

    if pending:
        chunk_count += retriever.index_chunks(pending)

    logger.info(
        "Done. Documents: %d | Chunks indexed: %d | Collection: %s",
        doc_count,
        chunk_count,
        settings.qdrant_collection,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed corpus and build Qdrant index")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed corpus files (default: data/processed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Chunk accumulation batch size before each upsert call (default: 50)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()
    build_index(Path(args.data_dir), batch_size=args.batch_size, clear=args.clear)


if __name__ == "__main__":
    main()
