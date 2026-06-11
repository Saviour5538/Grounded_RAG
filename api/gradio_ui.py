"""Gradio 6 UI for GroundedRAG — mounted on the FastAPI app.

Three tabs:
  Chat    — streaming Q&A via gr.ChatInterface, metrics / citations / sources
  Index   — upload & index corpus files with live progress
  Corpus  — browse indexed sources and BM25 status
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

import gradio as gr


# ── Visual helpers ─────────────────────────────────────────────────────────────

def _tier(score: float) -> str:
    if score >= 0.7: return "🟢"
    if score >= 0.4: return "🟡"
    return "🔴"


def _pct(v: float) -> str:
    return f"{round(v * 100)}%"


def _fmt_metrics(evt: dict) -> str:
    conf  = evt.get("confidence") or {}
    sig   = conf.get("signals") or {}
    score = conf.get("score")
    faith = evt.get("faithfulness")
    relev = evt.get("answer_relevancy")
    lat   = evt.get("latency_ms")
    model = (evt.get("model") or "").replace("models/", "")
    usage = evt.get("usage") or {}
    cache = evt.get("cache_hit", False)
    refs  = evt.get("reformulations") or []

    rows: list[str] = []
    if score is not None:
        rows.append(f"| Confidence | **{score:.3f}** | {_tier(score)} |")
    if faith is not None:
        rows.append(f"| Faithfulness | **{_pct(faith)}** | {_tier(faith)} |")
    if relev is not None:
        rows.append(f"| Answer Relevancy | **{_pct(relev)}** | {_tier(relev)} |")

    rs  = sig.get("reranker_score")
    rrf = sig.get("rrf_signal")
    tok = sig.get("token_overlap")
    if rs  is not None: rows.append(f"| Reranker Score | **{rs:.3f}** | {_tier(rs)} |")
    if rrf is not None: rows.append(f"| RRF Signal | **{rrf:.3f}** | {_tier(rrf)} |")
    if tok is not None: rows.append(f"| Token Overlap | **{tok:.3f}** | {_tier(tok)} |")

    if not rows:
        return "*No metrics available.*"

    lines = ["| Metric | Value | Grade |", "|--------|-------|-------|"] + rows

    perf: list[str] = []
    if lat:   perf.append(f"⚡ {lat:,} ms")
    if model: perf.append(f"`{model}`")
    if usage: perf.append(f"↑{usage.get('input_tokens',0)} ↓{usage.get('output_tokens',0)} tokens")
    if cache: perf.append("💙 cache hit")
    if perf:
        lines.append(f"\n*{' &nbsp;·&nbsp; '.join(perf)}*")
    if refs:
        lines.append(f"\n> ⟳ Reformulated: {' → '.join(refs)}")

    return "\n".join(lines)


def _fmt_citations(citations: dict | None) -> str:
    if not citations or not citations.get("sentences"):
        return "*No citation data.*"
    ok  = citations.get("verification_passed", False)
    unc = citations.get("uncited_count", 0)
    uns = citations.get("unsupported_count", 0)
    lines = [f"{'✅' if ok else '⚠️'} **{unc}** uncited &nbsp;·&nbsp; **{uns}** unsupported\n"]
    for s in citations["sentences"]:
        reason = s.get("reason", "")
        if s.get("supported") is True:
            bullet = "✅"
        elif reason in ("no_citation", "invalid_ref"):
            bullet = "❓"
        else:
            bullet = "❌"
        lines.append(f"{bullet} {s['text']}")
    return "\n".join(lines)


def _fmt_sources(chunks: list[dict]) -> str:
    if not chunks:
        return "*No sources retrieved.*"
    lines: list[str] = []
    for i, c in enumerate(chunks[:5], 1):
        src     = c.get("source") or c.get("doc_id") or "unknown"
        label   = Path(src).name
        rrf     = c.get("score", 0.0)
        rer     = c.get("reranker_score")
        scores  = f"RRF `{rrf:.4f}`"
        if rer is not None:
            scores += f" &nbsp;·&nbsp; Reranker `{rer:.2f}`"
        preview = (c.get("text") or "")[:250].replace("\n", " ")
        lines.append(f"**{i}. {label}** — {scores}\n> {preview}…\n")
    return "\n".join(lines)


# ── Main builder ───────────────────────────────────────────────────────────────

def build_demo(
    pipeline: Any,
    ingest_gen: Callable[[bytes, str, bool], Iterator[dict]],
) -> gr.Blocks:
    """Return the gr.Blocks instance to be mounted via gr.mount_gradio_app().

    theme and css are passed to mount_gradio_app() in main.py (Gradio 6 API).

    Parameters
    ----------
    pipeline   : RAGPipeline shared with FastAPI
    ingest_gen : callable(file_bytes, filename, clear) → iterator of progress dicts
    """

    # ── Chat response (streaming generator for gr.ChatInterface) ───────────────

    def respond(
        message: str,
        history: list[dict],
        threshold: float,
    ) -> Generator:
        """Yields (response_chunk, metrics_md, cite_md, src_md) tuples."""
        if not message.strip():
            yield "", "*Ask a question to see metrics.*", "", ""
            return

        # Gradio 6 ChatInterface history: list of {"role": "user"|"assistant", "content": str}
        pipeline_history: list[dict] = []
        pending_q: str | None = None
        for msg in (history or []):
            role    = msg.get("role", "")
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if role == "user":
                pending_q = content
            elif role == "assistant" and pending_q is not None:
                pipeline_history.append({"question": pending_q, "answer": content})
                pending_q = None

        partial  = ""
        meta_md  = "*Retrieving…*"
        cite_md  = ""
        src_md   = ""

        yield "▌", meta_md, "", ""

        try:
            for sse in pipeline.query_stream(
                message,
                history=pipeline_history[-5:] or None,
                confidence_threshold=float(threshold),
            ):
                if not sse.startswith("data: "):
                    continue
                try:
                    evt = json.loads(sse[6:])
                except Exception:
                    continue

                t = evt.get("type")

                if t == "retrieval":
                    n = len(evt.get("chunks", []))
                    yield "▌", f"*Retrieved {n} candidates — generating…*", "", ""

                elif t == "token":
                    partial += evt["text"]
                    yield partial + "▌", meta_md, "", ""

                elif t == "abstain":
                    ans     = evt.get("answer", "Insufficient evidence found.")
                    meta_md = _fmt_metrics(evt) or "*Abstained — confidence below threshold.*"
                    yield ans, meta_md, "*Abstained.*", ""
                    return

                elif t == "meta":
                    ans     = evt.get("answer", partial)
                    meta_md = _fmt_metrics(evt)
                    cite_md = _fmt_citations(evt.get("citations"))
                    src_md  = _fmt_sources(evt.get("chunks", []))
                    yield ans, meta_md, cite_md, src_md
                    return

                elif t == "error":
                    yield f"⚠️ Pipeline error: {evt.get('message')}", "", "", ""
                    return

        except Exception as exc:
            yield f"⚠️ {exc}", "", "", ""
            return

        yield partial, meta_md, cite_md, src_md

    # ── Upload (streaming generator) ───────────────────────────────────────────

    def do_upload(file, clear_existing: bool) -> Generator:
        if file is None:
            yield "⚠️ Please select a file first."
            return

        path      = Path(file.name)
        filename  = path.name
        try:
            file_bytes = path.read_bytes()
        except Exception as exc:
            yield f"❌ Could not read file: {exc}"
            return

        yield f"⏳ Uploading **{filename}** ({len(file_bytes)/1024:.0f} KB)…"

        try:
            for update in ingest_gen(file_bytes, filename, bool(clear_existing)):
                status   = update.get("status", "running")
                progress = update.get("progress", {})
                docs_p   = progress.get("docs_processed", 0)
                docs_t   = progress.get("docs_total", 1)
                chunks_n = progress.get("chunks_indexed", 0)
                error    = update.get("error")

                if error:
                    yield f"❌ **Error:** {error}"
                    return
                if status == "done":
                    yield (f"✅ **Done!** &nbsp; {docs_t} docs &nbsp;·&nbsp; {chunks_n:,} chunks indexed")
                    return
                pct = round(docs_p / max(docs_t, 1) * 100)
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                yield f"`{bar}` {pct}% &nbsp;·&nbsp; {docs_p}/{docs_t} docs &nbsp;·&nbsp; {chunks_n:,} chunks"
        except Exception as exc:
            yield f"❌ Ingest failed: {exc}"

    # ── Corpus stats ────────────────────────────────────────────────────────────

    def load_corpus() -> str:
        try:
            stats   = pipeline.retriever.corpus_stats()
            total   = stats.get("total_chunks", 0)
            uniq    = stats.get("unique_sources", 0)
            bm25    = stats.get("bm25_built", False)
            sources = stats.get("sources", [])
            err     = stats.get("error")

            lines = [
                f"**{total:,}** total chunks &nbsp;·&nbsp; **{uniq}** unique sources",
                "🟢 BM25 index ready" if bm25 else "⚪ BM25 not built yet (run a query first)",
            ]
            if err:
                lines.append(f"\n⚠️ {err}")
            if sources:
                lines.append("\n| Source | Chunks |")
                lines.append("|--------|--------|")
                for s in sources:
                    lines.append(f"| `{s['name']}` | {s['chunks']:,} |")
            else:
                lines.append("\n*No sources indexed — upload a file in the Index tab.*")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Could not load corpus stats: {exc}"

    def health_status() -> str:
        try:
            n = pipeline.retriever.collection_size()
            return f"🟢 Online &nbsp;·&nbsp; **{n:,}** chunks indexed"
        except Exception:
            return "🔴 Pipeline offline"

    # ── Build Blocks ────────────────────────────────────────────────────────────
    # Note: theme and css are passed to gr.mount_gradio_app() in main.py (Gradio 6)

    with gr.Blocks(title="GroundedRAG") as demo:

        gr.HTML("""
        <div style="display:flex;align-items:center;gap:14px;padding:16px 0 8px">
          <span style="font-size:36px;line-height:1">🧠</span>
          <div>
            <h1 style="margin:0;font-size:26px;font-weight:800;letter-spacing:-0.5px">
              Grounded<em style="color:#7c3aed">RAG</em>
            </h1>
            <p style="margin:2px 0 0;font-size:12px;color:#64748b">
              Hybrid retrieval &nbsp;·&nbsp; grounded answers &nbsp;·&nbsp;
              confidence-based abstention &nbsp;·&nbsp; Phase 8
            </p>
          </div>
          <div id="health-placeholder" style="margin-left:auto"></div>
        </div>
        <hr style="margin:0 0 12px;border:none;border-top:1px solid #e2e8f0">
        """)

        health_md = gr.Markdown(value="*checking…*")

        with gr.Tabs():

            # ── Tab 1: Chat ───────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):

                # Output panels (defined first so ChatInterface can reference them)
                with gr.Accordion("📊 Pipeline Metrics", open=True):
                    metrics_out = gr.Markdown(value="*Ask a question to see metrics.*")

                with gr.Row():
                    with gr.Accordion("🔗 Citations", open=False):
                        cite_out = gr.Markdown(value="*Citations appear here after answering.*")
                    with gr.Accordion("📄 Sources", open=False):
                        src_out = gr.Markdown(value="*Retrieved sources appear here after answering.*")

                # ChatInterface — threshold slider lives in the built-in accordion
                gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                            label="Abstention Threshold",
                            info="Lower = answer more  |  Higher = abstain more",
                        ),
                    ],
                    additional_inputs_accordion=gr.Accordion("⚙️ Settings", open=False),
                    additional_outputs=[metrics_out, cite_out, src_out],
                    submit_btn="Ask ↵",
                    stop_btn="Stop",
                    examples=[
                        ["What is this document about?", 0.3],
                        ["Summarise the key findings.", 0.3],
                        ["What evidence is given for the main claim?", 0.3],
                    ],
                    fill_height=True,
                )

            # ── Tab 2: Index ──────────────────────────────────────────────────
            with gr.Tab("📂 Index Documents"):
                gr.Markdown("""### Upload & Index a Corpus
Drag and drop a file or click to browse. The pipeline will chunk, embed, and index it automatically.

**Supported formats** &nbsp;·&nbsp; **PDF** &nbsp;·&nbsp; **JSONL** (one record per line with a `text` field)
&nbsp;·&nbsp; **JSON** (array with `text`/`abstract`/`content`) &nbsp;·&nbsp; **CSV** (with `text` column)
""")
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input   = gr.File(
                            label="Drop file here or click to browse",
                            file_types=[".pdf", ".json", ".jsonl", ".csv"],
                        )
                        clear_idx_cb = gr.Checkbox(
                            label="Clear existing index before uploading",
                            value=False,
                            info="Deletes all previously indexed chunks from Pinecone.",
                        )
                        upload_btn   = gr.Button("Upload & Index", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        upload_status = gr.Markdown(value="")

                upload_btn.click(
                    do_upload,
                    inputs=[file_input, clear_idx_cb],
                    outputs=[upload_status],
                )

            # ── Tab 3: Corpus Browser ─────────────────────────────────────────
            with gr.Tab("🗂️ Corpus Browser"):
                gr.Markdown("""### Indexed Sources
Per-source chunk breakdown from the BM25 index.
The BM25 index is built lazily on the first query — run a query first if the list is empty.
""")
                corpus_btn = gr.Button("↺ Refresh", variant="secondary", size="sm")
                corpus_md  = gr.Markdown(value="*Click Refresh to load corpus stats.*")
                corpus_btn.click(load_corpus, outputs=[corpus_md])

        demo.load(health_status, outputs=[health_md])

    return demo
