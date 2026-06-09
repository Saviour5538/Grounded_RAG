"""FastAPI application for the Grounded RAG service.

Run with:
    uvicorn api.main:app --reload

Endpoints:
    GET  /             — web UI (upload + query)
    GET  /health       — liveness check
    POST /query        — answer a question from the indexed corpus
    POST /ingest       — upload a JSON/JSONL file and index it
    GET  /ingest/status/{job_id}  — poll indexing progress
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_MAX_HISTORY_TURNS = 5   # how many prior Q&A turns to keep per session

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Ensure project root is on the path when running directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from grounded_rag.ingest.chunker import chunk_document
from grounded_rag.ingest.dedup import dedup_documents
from grounded_rag.ingest.loader import Document, _normalize, load_jsonl, load_pdf
from grounded_rag.ingest.metadata import enrich_document
from grounded_rag.pipeline import RAGPipeline
from grounded_rag.retrieval.dense import DenseRetriever, EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None

# ── In-memory job registry ────────────────────────────────────────────────────
# job_id → {status, progress, error, started_at, finished_at}
_jobs: dict[str, dict[str, Any]] = {}

# ── Session memory store ──────────────────────────────────────────────────────
# session_id → [{"question": str, "answer": str}, ...]  (last N turns)
_sessions: dict[str, list[dict[str, Any]]] = {}

_UI_HTML = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Initialising RAG pipeline...")
    _pipeline = RAGPipeline()
    logger.info("Pipeline ready")
    yield
    _pipeline = None


app = FastAPI(
    title="Grounded RAG",
    description="RAG pipeline with grounded, citable answers and confidence-based abstention.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    session_id: str | None = None   # omit for stateless; provide to use session memory


class QueryResponse(BaseModel):
    answer: str
    question: str
    chunks: list[dict[str, Any]]
    model: str | None = None
    usage: dict[str, Any] | None = None
    confidence: dict[str, Any] | None = None
    citations: dict[str, Any] | None = None
    latency_ms: int | None = None
    cache_hit: bool = False
    reformulations: list[str] = []
    session_id: str | None = None
    history_turns: int = 0   # how many prior turns informed this answer


# ── Ingest helpers ────────────────────────────────────────────────────────────

def _load_upload_bytes(file_bytes: bytes, filename: str):
    """Parse uploaded bytes into Document objects.

    Dispatches by extension: .pdf → load_pdf, .json → JSON array,
    .jsonl/.csv → temp-file loaders.
    """
    suffix = Path(filename).suffix.lower()

    # PDF: write raw bytes to a temp file and use load_pdf
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            yield from load_pdf(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    text = file_bytes.decode("utf-8")

    # Try JSON array first if .json extension or starts with '['
    if suffix == ".json" or text.lstrip().startswith("["):
        try:
            items = json.loads(text)
            if isinstance(items, list):
                _TEXT_FIELDS = ("text", "content", "abstract", "body", "passage")
                for obj in items:
                    body = _normalize(next((obj[k] for k in _TEXT_FIELDS if obj.get(k)), ""))
                    if not body:
                        continue
                    raw_id = obj.get("id") or obj.get("_id")
                    doc_id = str(raw_id) if raw_id else hashlib.sha256(body.encode()).hexdigest()[:16]
                    metadata = {k: v for k, v in obj.items() if k not in {*_TEXT_FIELDS, "id", "_id"}}
                    yield Document(id=doc_id, text=body, source=filename, metadata=metadata)
                return
        except json.JSONDecodeError:
            pass  # fall through to JSONL

    # JSONL / CSV: write to a temp file and use existing loaders
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        yield from load_jsonl(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_ingest_job(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    clear_existing: bool,
) -> None:
    """Synchronous background task: parse → chunk → embed → index."""
    job = _jobs[job_id]
    job["status"] = "running"

    try:
        # ── Load documents ────────────────────────────────────────────────────
        raw_docs = list(_load_upload_bytes(file_bytes, filename))
        if not raw_docs:
            raise ValueError("No documents found in the uploaded file. "
                             "Ensure each record has a 'text', 'content', or 'abstract' field.")

        deduped = list(dedup_documents(iter(raw_docs)))
        job["progress"]["docs_total"] = len(deduped)
        logger.info("[job %s] Loaded %d docs (%d after dedup)", job_id, len(raw_docs), len(deduped))

        # ── Set up retriever ──────────────────────────────────────────────────
        embed_key = (
            settings.gemini_api_key if settings.embedding_provider == "gemini"
            else settings.openai_api_key
        )
        embedder = EmbeddingModel(
            model_name=settings.embedding_model,
            provider=settings.embedding_provider,
            api_key=embed_key,
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
            if clear_existing:
                retriever.delete_index()
                logger.info("[job %s] Deleted Pinecone index", job_id)
        else:
            retriever = DenseRetriever(
                collection=settings.qdrant_collection,
                embedding_model=embedder,
                dim=settings.embedding_dim,
                qdrant_url=settings.qdrant_url,
                qdrant_mode=settings.qdrant_mode,
                qdrant_local_path=settings.qdrant_local_path,
                qdrant_api_key=settings.qdrant_api_key,
            )
            if clear_existing:
                existing = {c.name for c in retriever.client.get_collections().collections}
                if settings.qdrant_collection in existing:
                    retriever.client.delete_collection(settings.qdrant_collection)
                    logger.info("[job %s] Cleared existing collection", job_id)

        retriever.ensure_collection()

        # ── Chunk → embed → index in batches ──────────────────────────────────
        BATCH = 50
        pending = []
        docs_done = 0

        for doc in deduped:
            enriched = enrich_document(doc)
            chunks = chunk_document(enriched, settings.chunk_size, settings.chunk_overlap)
            pending.extend(chunks)
            docs_done += 1

            if len(pending) >= BATCH:
                indexed = retriever.index_chunks(pending)
                job["progress"]["chunks_indexed"] += indexed
                job["progress"]["docs_processed"] = docs_done
                logger.info("[job %s] %d/%d docs | %d chunks",
                            job_id, docs_done, len(deduped),
                            job["progress"]["chunks_indexed"])
                pending = []

        if pending:
            indexed = retriever.index_chunks(pending)
            job["progress"]["chunks_indexed"] += indexed
            job["progress"]["docs_processed"] = docs_done

        job["status"] = "done"
        job["finished_at"] = time.time()
        logger.info("[job %s] Done — %d docs, %d chunks indexed",
                    job_id, docs_done, job["progress"]["chunks_indexed"])

        # Invalidate the pipeline's BM25 index so the next query rebuilds it
        # from the updated Pinecone corpus (dense retriever is always live).
        if _pipeline is not None:
            _pipeline.retriever.sparse.invalidate()

    except Exception as exc:
        logger.exception("[job %s] Failed: %s", job_id, exc)
        job["status"] = "error"
        job["error"] = str(exc)
        job["finished_at"] = time.time()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    if _UI_HTML.exists():
        return _UI_HTML.read_text(encoding="utf-8")
    return HTMLResponse("<h2>UI not found — run with api/static/index.html present.</h2>")


@app.get("/health")
async def health():
    size = 0
    cache_stats = {}
    if _pipeline is not None:
        try:
            size = _pipeline.retriever.collection_size()
        except Exception:
            pass
        if _pipeline.cache:
            cache_stats = _pipeline.cache.stats()
    return {"status": "ok", "indexed_chunks": size, "cache": cache_stats}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    session_id = request.session_id
    history: list[dict[str, Any]] = []

    if session_id:
        history = list(_sessions.get(session_id, []))   # copy so pipeline can't mutate store

    result = _pipeline.query(request.question, history=history or None)

    # Persist this turn into session memory
    if session_id:
        if session_id not in _sessions:
            _sessions[session_id] = []
        _sessions[session_id].append({
            "question": request.question,
            "answer":   result["answer"],
        })
        _sessions[session_id] = _sessions[session_id][-_MAX_HISTORY_TURNS:]

    return QueryResponse(**result, session_id=session_id, history_turns=len(history))


@app.delete("/session/{session_id}", include_in_schema=False)
async def clear_session(session_id: str):
    """Clear a session's conversation history (called by 'New conversation' button)."""
    _sessions.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}


@app.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clear_existing: bool = Form(False),
):
    """Upload a JSON or JSONL corpus file and index it in the background."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".json", ".jsonl", ".csv", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .json, .jsonl, .csv, or .pdf file.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "progress": {"docs_total": 0, "docs_processed": 0, "chunks_indexed": 0},
        "error": None,
        "filename": file.filename,
        "started_at": time.time(),
        "finished_at": None,
    }

    background_tasks.add_task(
        _run_ingest_job, job_id, file_bytes, file.filename, clear_existing
    )

    return {"job_id": job_id, "message": f"Indexing started for '{file.filename}'"}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    prog = job["progress"]
    total = prog["docs_total"] or 1
    pct = round(prog["docs_processed"] / total * 100)

    elapsed = time.time() - job["started_at"]
    eta_sec = None
    if job["status"] == "running" and prog["docs_processed"] > 0:
        rate = prog["docs_processed"] / elapsed
        remaining = total - prog["docs_processed"]
        eta_sec = int(remaining / rate) if rate > 0 else None

    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename"),
        "progress": prog,
        "percent": pct,
        "eta_seconds": eta_sec,
        "error": job["error"],
        "elapsed_seconds": int(elapsed),
    }


@app.get("/ingest/jobs")
async def list_jobs():
    """List all ingest jobs (most recent first)."""
    return sorted(
        [
            {
                "job_id": jid,
                "status": j["status"],
                "filename": j.get("filename"),
                "chunks_indexed": j["progress"]["chunks_indexed"],
                "started_at": j["started_at"],
            }
            for jid, j in _jobs.items()
        ],
        key=lambda x: x["started_at"],
        reverse=True,
    )
