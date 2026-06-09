# Grounded RAG

A production-minded **Retrieval-Augmented Generation pipeline** whose defining
features are **grounded, citable answers** and **knowing when not to answer**.

Every component is built from first principles — no LangChain, no LlamaIndex.
Every improvement is justified by an eval number. The commit history reads as
a log of measured gains.

---

## What makes this different

| Feature | Demo chatbot | This project |
|---|---|---|
| Retrieval | Dense only | BM25 + dense embeddings + Reciprocal Rank Fusion |
| Ranking | Cosine similarity | Gemini cross-encoder reranker (top 50 → top 5) |
| Confidence | None | Weighted reranker × RRF × token-overlap signal |
| Uncertain queries | Hallucinate | Abstain — *"Insufficient evidence found."* |
| Source attribution | None | Every claim traced to a chunk with `[N]` refs |
| Claim verification | None | Batched Gemini pass flags uncited / unsupported sentences |
| Agentic recovery | None | Query reformulation + re-retrieval before final abstention |
| Conversation | Stateless | Session memory — up to 5 prior turns injected into prompt |
| Evaluation | Vibe check | RAGAS + hallucination rate + **abstention-correctness** |
| Observability | None | Per-request JSONL traces + optional Langfuse dashboard |
| Corpus upload | Scripts only | Drag-and-drop UI — PDF / JSONL / JSON / CSV |

---

## Measured improvements

All evals run on 50 SciFact Q&A pairs (biomedical claim verification corpus).

| Phase | What changed | Faithfulness | Answer Relevancy | Hallucination | Abstention |
|---|---|---|---|---|---|
| 2 — Baseline | Dense retrieval only, no rerank | 0.028 | 0.434 | 8.0 % | 0 % |
| 3 — Hybrid retrieval | BM25 + dense + RRF fusion | 0.040 | 0.462 | 6.4 % | 6 % |
| 4+5 — Reranker + confidence gate | Gemini reranker, abstention | 0.068 | 0.485 | **0.0 %** | 4 % |

Key result: the reranker + confidence gate **eliminated measured hallucination**
on the eval set while keeping 96 % of answerable questions answered.

---

## Architecture

Ten components in data-flow order:

```
User query
  │
  ├─ [9] Cache check — SHA-256 keyed LRU (skipped for session queries)
  │        HIT → return instantly
  │        MISS ↓
  │
  ├─ [2] Hybrid Retrieval
  │        ├── BM25 sparse (rank_bm25, in-memory, rebuilt after each upload)
  │        └── Dense (Gemini gemini-embedding-001, 3072-dim → Pinecone ANN)
  │        └── Reciprocal Rank Fusion (k=60) → top 50 candidates
  │
  ├─ [3] Gemini Reranker
  │        Single batched prompt → relevance scores 0-10 → top 5 chunks
  │
  ├─ [4] Confidence Scorer
  │        score = 0.55 × reranker_signal
  │              + 0.25 × rrf_signal
  │              + 0.20 × token_overlap
  │
  ├─ [7] Abstention gate ── score < 0.3?
  │        YES → [8] Agentic retry
  │               attempt 1: expand query (add synonyms + context)
  │               attempt 2: decompose to core sub-claim
  │               still low? → "Insufficient evidence found."
  │        NO  ↓
  │
  ├─ [5] Constrained generation (Gemini Flash)
  │        System prompt: answer ONLY from numbered passages, cite [N]
  │        Session history (last 5 turns) prepended when session active
  │
  ├─ [6] Citation verification
  │        Parse [N] refs → batch Gemini call → flag uncited / unsupported
  │
  └─ [10] Trace log → JSONL file + optional Langfuse
```

Ingestion pipeline (runs in background after each upload):

```
Uploaded file (PDF / JSONL / JSON / CSV)
  │
  ├─ [1] Loader — NFKC normalize, whitespace collapse
  │        PDF: header/footer strip, hyphen-break fix (PyMuPDF)
  │
  ├─ Dedup — SHA-256 fingerprint (case-insensitive)
  ├─ Chunker — recursive split, 512 chars / 64 overlap
  ├─ Metadata enrichment — word count, ingest timestamp
  ├─ Embed — Gemini gemini-embedding-001 (batched, rate-limit aware)
  └─ Pinecone upsert → BM25 index invalidated → rebuilds on next query
```

---

## Quick start

### Prerequisites

- Python 3.11+
- [Gemini API key](https://aistudio.google.com/) (free tier)
- [Pinecone account](https://app.pinecone.io/) (free Serverless tier — 2 GB forever)

### 1. Clone and install

```bash
git clone <repo>
cd grounded-rag
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Minimum required fields in `.env`:

```env
GEMINI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=grounded-rag
```

### 3. Start the server

```bash
uvicorn api.main:app --reload
```

Open **http://localhost:8000** — the web UI handles everything from here.

### 4. Index a corpus

**Option A — Web UI (recommended):**
Upload any `.pdf`, `.jsonl`, `.json`, or `.csv` file via the "Index Corpus" panel.
Progress is tracked live with docs/chunks/ETA.

**Option B — SciFact benchmark corpus (for eval):**
```bash
python scripts/ingest_corpus.py   # downloads ~5 k biomedical abstracts + qrels
python scripts/build_index.py     # embeds → Pinecone (run once)
```

### 5. Ask questions

Via UI, or directly:

```bash
# Stateless query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does cardiopulmonary fitness reduce mortality?"}'

# Session query (maintains conversation context)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What about in elderly patients?", "session_id": "my-session-1"}'
```

Example response:

```json
{
  "answer": "High cardiopulmonary fitness is associated with lower all-cause mortality [1][2].",
  "confidence": {
    "score": 0.741,
    "signals": { "reranker_score": 0.82, "rrf_signal": 0.71, "token_overlap": 0.43 },
    "abstain": false
  },
  "citations": {
    "verification_passed": true,
    "uncited_count": 0,
    "unsupported_count": 0
  },
  "latency_ms": 1840,
  "cache_hit": false,
  "session_id": "my-session-1",
  "history_turns": 0
}
```

Abstention example (NOT_ENOUGH_INFO query):
```json
{
  "answer": "Insufficient evidence found.",
  "confidence": { "score": 0.18, "abstain": true }
}
```

---

## Running the eval harness

```bash
# Server must be running first
uvicorn api.main:app --reload &

# Full eval — 50 SciFact samples
python evals/run_evals.py

# Faster run — skip RAGAS, custom metrics only
python evals/run_evals.py --skip-ragas --n-samples 20
```

Sample output:
```
════════════════════════════════════════════════════════
  GROUNDED RAG — EVAL REPORT
════════════════════════════════════════════════════════

  RAGAS Metrics  (higher = better)
  ────────────────────────────────────────────
  faithfulness                         0.068  █
  answer_relevancy                     0.485  █████████
  context_precision                    0.621  ████████████

  Custom Metrics
  ────────────────────────────────────────────
  abstention_rate                      0.040
  hallucination_rate                   0.000
  avg_confidence                       0.641

  Abstention Correctness  (crown jewel)
  ────────────────────────────────────────────
  abstention_precision                 0.875  █████████████████
  abstention_recall                    0.700  ██████████████
  abstention_f1                        0.778  ███████████████
  false_negative_rate                  0.021
════════════════════════════════════════════════════════
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness check + cache stats + chunk count |
| `POST` | `/query` | Ask a question (`session_id` optional for memory) |
| `DELETE` | `/session/{id}` | Clear a conversation session |
| `POST` | `/ingest` | Upload corpus file (background job) |
| `GET` | `/ingest/status/{job_id}` | Poll indexing progress |
| `GET` | `/ingest/jobs` | List all ingest jobs |

**`POST /query` body:**
```json
{
  "question": "string",
  "session_id": "optional-string"
}
```

---

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Embeddings | `gemini-embedding-001` (3072-dim) | RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types, no local model |
| Vector DB | Pinecone Serverless | Free tier (2 GB forever), no infrastructure to run |
| Sparse retrieval | `rank-bm25` (BM25Okapi, in-memory) | Zero infra, invalidated and rebuilt after each upload |
| Reranker | Gemini Flash (zero-shot cross-encoder prompt) | No separate reranker model to host |
| LLM | `gemini-2.5-flash` | Fast, cheap, `thinking_budget=0` for aux calls |
| PDF parsing | PyMuPDF | Header/footer detection, hyphen-break fix |
| Eval | RAGAS-equivalent (Gemini as judge) + custom metrics | No RAGAS dependency, full control over scoring logic |
| API | FastAPI | Async, auto OpenAPI spec |
| Observability | JSONL traces + Langfuse (optional) | Zero-dependency default; rich dashboard opt-in |

---

## Repo structure

```
grounded-rag/
├── src/grounded_rag/
│   ├── ingest/          loader.py  dedup.py  chunker.py  metadata.py
│   ├── retrieval/       dense.py  sparse.py  hybrid.py  reranker.py
│   │                    pinecone_retriever.py  reformulator.py
│   ├── confidence/      scorer.py
│   ├── generation/      generator.py  citations.py
│   ├── cache/           query_cache.py
│   ├── observability/   tracer.py
│   └── pipeline.py
├── api/
│   ├── main.py          FastAPI app, session store, ingest background jobs
│   └── static/index.html  Web UI (dark theme, confidence bar, citation panel)
├── evals/
│   ├── metrics.py       faithfulness, hallucination, abstention_correctness
│   ├── run_evals.py     harness runner
│   └── datasets/        scifact_qa.py (with qrels for NEI labels)
├── scripts/
│   ├── ingest_corpus.py  download SciFact corpus + qrels
│   └── build_index.py    CLI embed + index
├── tests/
│   ├── test_confidence_scorer.py   (18 tests)
│   └── test_citations.py           (23 tests)
├── config/settings.py   pydantic-settings, all config via .env
├── requirements.txt
├── .env.example
└── docker-compose.yml   optional Qdrant (not needed with Pinecone)
```

---

## Build sequence

```
Phase 1  Vertical slice — naive dense RAG end-to-end (baseline)
Phase 2  Eval harness — SciFact dataset, RAGAS + custom metrics
         faithfulness 0.028 · hallucination 8.0%
Phase 3  Hybrid retrieval — BM25 + dense + RRF
         hallucination → 6.4%
Phase 4  Gemini cross-encoder reranker (top 50 → top 5)
Phase 5  Confidence scoring + abstention gate
         hallucination → 0.0% · abstention-correctness eval added
Phase 6  Constrained generation prompt + citation verification
Phase 7  LRU query cache + JSONL observability + Langfuse
Phase 8  Agentic re-retrieval — expand / decompose reformulation
+        Session memory (5-turn conversation context)
+        PDF ingestion (PyMuPDF, header/footer stripping)
+        Pinecone Serverless migration (from local Qdrant)
```

---

## Configuration reference

Key `.env` settings:

```env
# Required
GEMINI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=grounded-rag

# Retrieval (tuned defaults)
RETRIEVAL_TOP_K=50        # ANN candidates → reranker
RERANKER_TOP_N=5          # chunks kept after reranking

# Confidence / abstention
CONFIDENCE_THRESHOLD=0.3  # below this → abstain (or retry)

# Agentic retry
AGENTIC_RETRY_ENABLED=true
AGENTIC_MAX_RETRIES=2

# Cache
CACHE_ENABLED=true
CACHE_MAX_SIZE=512

# Observability (optional)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```
