# Grounded RAG

A production-minded **Retrieval-Augmented Generation pipeline** whose defining
features are **grounded, citable answers** and **knowing when not to answer**.

Every improvement is justified by an eval number. The commit history reads as
a log of measured gains, not a wishlist.

---

## What makes this different

| Feature | Demo chatbot | This project |
|---------|-------------|-------------|
| Retrieval | Dense only | BM25 + dense + Reciprocal Rank Fusion |
| Ranking | cosine similarity | Gemini cross-encoder reranker |
| Confidence | none | weighted reranker × RRF × token-overlap signal |
| Uncertain queries | hallucinate | abstain — *"Insufficient evidence found."* |
| Source attribution | none | every claim traced to a chunk with [N] refs |
| Claim verification | none | batched Gemini pass flags uncited/unsupported sentences |
| Agentic recovery | none | query reformulation + re-retrieval before final abstention |
| Observability | none | per-request JSONL traces + optional Langfuse |
| Evaluation | vibe check | RAGAS + custom hallucination + abstention-correctness |

---

## Measured improvements

All evals run on 50 SciFact Q&A pairs (biomedical claim verification corpus).

| Phase | What changed | Faithfulness | Answer Relevancy | Hallucination | Abstention |
|-------|-------------|-------------|-----------------|--------------|------------|
| 2 — Baseline | Dense retrieval, no rerank | 0.028 | 0.434 | 8.0% | 0% |
| 3 — Hybrid retrieval | BM25 + dense + RRF | 0.040 | 0.462 | 6.4% | 6% |
| 4+5 — Reranker + confidence | Gemini reranker, abstention gate | 0.068 | 0.485 | **0.0%** | 4% |

Key result: the reranker + confidence gate **eliminated measured hallucination**
on the eval set while keeping 96% of answerable questions answered.

---

## Architecture

Ten components in data-flow order:

```
Query
  │
  ├─ 9. Cache check (LRU, SHA-256 keyed)
  │
  ├─ 2. Hybrid Retrieval ── BM25/sparse ─────────┐
  │        └──────────── dense (Gemini embed) ───── RRF fusion → top-20
  │
  ├─ 3. Gemini Reranker ── scores 20 candidates, keeps top-5
  │
  ├─ 4. Confidence Scorer
  │       0.55 × reranker_score
  │     + 0.25 × rrf_signal
  │     + 0.20 × token_overlap
  │
  ├─ 7. Abstention gate ── score < 0.3?
  │         YES → 8. Agentic retry (expand → decompose, up to 2x)
  │               still low? → "Insufficient evidence found."
  │         NO  ↓
  │
  ├─ 5. Constrained generation (Gemini Flash, context-only prompt)
  │
  ├─ 6. Citation verification
  │       parse [N] refs → batch Gemini call → flag uncited/unsupported
  │
  └─ 10. Trace log (JSONL + optional Langfuse)
```

Ingestion pipeline (run once):
```
Raw docs → 1. Loader → Dedup → Chunker (512 tok, 64 overlap) → Metadata
         → Gemini embed (3072-dim) → Qdrant (dense + sparse vectors)
```

---

## Quick start

### Prerequisites

- Python 3.11+
- A Gemini API key (free tier works for small corpora)

### 1. Install

```bash
git clone <repo>
cd grounded-rag
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set GEMINI_API_KEY
```

### 3. Index a corpus

The repo ships a SciFact downloader script:

```bash
python scripts/ingest_corpus.py   # downloads SciFact (~5 k docs) to data/raw/
python scripts/build_index.py     # chunks → embeds → stores in Qdrant (local file mode)
```

Or upload your own `.json` / `.jsonl` file via the API after starting the server.

### 4. Start the server

```bash
uvicorn api.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the web UI, or use the
API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does high cardiopulmonary fitness increase mortality?"}'
```

Example response:

```json
{
  "answer": "High cardiopulmonary fitness is associated with lower mortality risk [1].",
  "confidence": {"score": 0.72, "abstain": false},
  "citations": {"uncited_count": 0, "unsupported_count": 0, "verification_passed": true},
  "cache_hit": false,
  "latency_ms": 1840
}
```

---

## Running the eval harness

```bash
# Full eval — 50 samples, calls the API server (must be running)
python evals/run_evals.py

# Results saved to evals/results/<timestamp>.json
```

The harness measures:
- **RAGAS** — faithfulness, answer relevancy, context precision
- **Hallucination rate** — sentences not grounded in retrieved chunks
- **Abstention rate** — fraction of queries that triggered the confidence gate
- **Avg confidence / reranker score** — retrieval quality signals

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Web UI |
| `GET`  | `/health` | Liveness + cache stats |
| `POST` | `/query` | Ask a question |
| `POST` | `/ingest` | Upload corpus file (background job) |
| `GET`  | `/ingest/status/{job_id}` | Poll indexing progress |
| `GET`  | `/ingest/jobs` | List all ingest jobs |

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Embeddings | `gemini-embedding-001` (3072-dim) | Matches LLM provider, no local model |
| Vector DB | Qdrant (local file mode) | Native sparse+dense support; no Docker needed for dev |
| Reranker | Gemini Flash (zero-shot cross-encoder prompt) | No separate reranker model to host |
| LLM | `gemini-2.5-flash` | Fast, cheap, thinking budget configurable |
| Eval | RAGAS + custom metrics | Industry-standard + domain-specific checks |
| API | FastAPI | Async, OpenAPI spec auto-generated |
| Observability | JSONL traces + Langfuse (optional) | Zero-dependency default, rich dashboard opt-in |

---

## Repo structure

```
grounded-rag/
├── src/grounded_rag/
│   ├── ingest/          loader, dedup, chunker, metadata
│   ├── retrieval/       dense, sparse, hybrid (RRF), reranker, reformulator
│   ├── confidence/      scorer (abstention gate)
│   ├── generation/      generator, citations (verification)
│   ├── cache/           query_cache (LRU)
│   ├── observability/   tracer (JSONL + Langfuse)
│   └── pipeline.py      end-to-end orchestrator
├── api/main.py          FastAPI app + ingest background jobs
├── evals/               metrics, run_evals, datasets, results/
├── scripts/             ingest_corpus.py, build_index.py
├── config/settings.py   pydantic-settings, all config from .env
├── tests/               unit tests (confidence scorer, citation verifier, chunker)
└── docker-compose.yml   Qdrant server (optional, local mode is default)
```

---

## Build sequence followed

```
Phase 1  Vertical slice: naive dense RAG end-to-end
Phase 2  Eval harness baseline — faithfulness 0.028, hallucination 8%
Phase 3  Hybrid retrieval (BM25 + dense + RRF) — hallucination → 6.4%
Phase 4  Cross-encoder reranking (Gemini)
Phase 5  Confidence scoring + abstention gate — hallucination → 0%
Phase 6  Constrained generation prompt + citation verification
Phase 7  LRU cache + JSONL observability tracing
Phase 8  Agentic re-retrieval (query reformulation on low confidence)
```
