"""Gradio 6 UI for Verity — grounded RAG with cited truth.

Layout
  💬 Chat    — sidebar (settings + pipeline) · chatbot · metric cards
  📂 Index   — file upload with live progress bar
  🗂️ Corpus  — per-source breakdown table
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

import gradio as gr


# ── Color helpers ──────────────────────────────────────────────────────────────

def _grade(v: float) -> tuple[str, str]:
    """Returns (label, hex_color) for a 0-1 score."""
    if v >= 0.7:
        return "HIGH",   "#059669"
    elif v >= 0.4:
        return "MED",    "#d97706"
    return "LOW",        "#dc2626"


def _pct(v: float) -> str:
    return f"{round(v * 100)}%"


# ── HTML component builders ────────────────────────────────────────────────────

def _metric_card(label: str, value: str, grade_label: str, grade_color: str) -> str:
    return (
        f'<div style="flex:1;min-width:92px;background:linear-gradient(145deg,#faf7ff,#ede9fe);'
        f'border:1px solid #ddd6fe;border-radius:12px;padding:14px 10px;text-align:center;'
        f'box-shadow:0 2px 8px rgba(124,58,237,0.07)">'
        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;'
        f'color:#9ca3af;margin-bottom:7px">{label}</div>'
        f'<div style="font-size:21px;font-weight:900;color:{grade_color};line-height:1">{value}</div>'
        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;'
        f'color:{grade_color};margin-top:6px;opacity:.9">{grade_label}</div>'
        f'</div>'
    )


def _fmt_metrics_html(evt: dict) -> str:
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
    rs    = sig.get("reranker_score")
    rrf   = sig.get("rrf_signal")
    tok   = sig.get("token_overlap")

    cards = ""
    if score is not None:
        g, gc = _grade(score);  cards += _metric_card("Confidence",    f"{score:.3f}", g, gc)
    if faith is not None:
        g, gc = _grade(faith);  cards += _metric_card("Faithfulness",  _pct(faith),   g, gc)
    if relev is not None:
        g, gc = _grade(relev);  cards += _metric_card("Relevancy",     _pct(relev),   g, gc)
    if rs   is not None:
        g, gc = _grade(rs);     cards += _metric_card("Reranker",      f"{rs:.3f}",   g, gc)
    if rrf  is not None:
        g, gc = _grade(rrf);    cards += _metric_card("RRF Signal",    f"{rrf:.3f}",  g, gc)
    if tok  is not None:
        g, gc = _grade(tok);    cards += _metric_card("Token Overlap", f"{tok:.3f}",  g, gc)

    if not cards:
        return _EMPTY_METRICS

    perf: list[str] = []
    if lat:   perf.append(f"⚡ {lat:,} ms")
    if model: perf.append(f"🤖 {model}")
    if usage: perf.append(f"↑{usage.get('input_tokens',0)} ↓{usage.get('output_tokens',0)} tok")
    if cache: perf.append("💙 cache hit")

    perf_html = ""
    if perf:
        perf_html = (
            '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;'
            'font-size:11px;color:#9ca3af">'
            + " &nbsp;·&nbsp; ".join(perf)
            + "</div>"
        )

    ref_html = ""
    if refs:
        ref_html = (
            '<div style="margin-top:8px;background:#f5f3ff;border:1px solid #ddd6fe;'
            'border-radius:8px;padding:7px 12px;font-size:12px;color:#5b21b6">'
            f'⟳ Reformulated: {" → ".join(refs)}</div>'
        )

    return (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;padding:4px 0">{cards}</div>'
        + perf_html + ref_html
    )


def _fmt_citations_html(citations: dict | None) -> str:
    if not citations or not citations.get("sentences"):
        return _EMPTY_CITE

    ok  = citations.get("verification_passed", False)
    unc = citations.get("uncited_count", 0)
    uns = citations.get("unsupported_count", 0)
    header_text   = "All claims cited & supported" if ok else f"{unc} uncited · {uns} unsupported"
    header_color  = "#065f46"  if ok else "#92400e"
    header_bg     = "#ecfdf5"  if ok else "#fffbeb"
    header_border = "#6ee7b7"  if ok else "#fcd34d"
    icon = "✅" if ok else "⚠️"

    rows = ""
    for s in citations["sentences"]:
        reason = s.get("reason", "")
        if s.get("supported") is True:
            sicon, sbg, sborder = "✅", "#f0fdf4", "#bbf7d0"
        elif reason in ("no_citation", "invalid_ref"):
            sicon, sbg, sborder = "❓", "#f9fafb", "#e5e7eb"
        else:
            sicon, sbg, sborder = "❌", "#fff1f2", "#fecaca"
        text = s["text"].replace("<", "&lt;").replace(">", "&gt;")
        rows += (
            f'<div style="background:{sbg};border:1px solid {sborder};border-radius:8px;'
            f'padding:8px 12px;margin:5px 0;font-size:13px;line-height:1.5;color:#374151">'
            f'{sicon} {text}</div>'
        )

    return (
        f'<div style="background:{header_bg};border:1px solid {header_border};border-radius:10px;'
        f'padding:10px 14px;margin-bottom:10px;font-size:13px;font-weight:600;color:{header_color}">'
        f'{icon} {header_text}</div>' + rows
    )


def _fmt_sources_html(chunks: list[dict]) -> str:
    if not chunks:
        return _EMPTY_SRC

    cards = ""
    for i, c in enumerate(chunks[:5], 1):
        src     = c.get("source") or c.get("doc_id") or "?"
        label   = Path(src).name
        rrf     = c.get("score", 0.0)
        rer     = c.get("reranker_score")
        sc_str  = f"RRF {rrf:.4f}"
        if rer is not None:
            sc_str += f" &nbsp;·&nbsp; Reranker {rer:.3f}"
        preview = (c.get("text") or "")[:240].replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")
        cards += (
            f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-left:3px solid #7c3aed;'
            f'border-radius:8px;padding:12px 14px;margin:6px 0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px">'
            f'<span style="font-weight:700;font-size:13px;color:#1f2937">'
            f'<span style="background:#ede9fe;color:#5b21b6;border-radius:5px;padding:1px 7px;'
            f'font-size:10px;font-weight:800;margin-right:7px">{i}</span>{label}</span>'
            f'<span style="font-size:11px;color:#9ca3af;font-family:monospace">{sc_str}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#6b7280;line-height:1.6;'
            f'border-top:1px solid #f0f0f0;padding-top:7px">{preview}…</div>'
            f'</div>'
        )
    return f'<div style="padding:4px 0">{cards}</div>'


def _health_html(n: int) -> str:
    return (
        '<div style="padding:18px 0;display:flex;justify-content:flex-end;align-items:center">'
        '<span style="display:inline-flex;align-items:center;gap:7px;background:#ecfdf5;'
        'border:1px solid #6ee7b7;border-radius:20px;padding:5px 14px;font-size:12px;'
        'font-weight:600;color:#065f46">'
        '<span style="width:7px;height:7px;background:#22c55e;border-radius:50%;'
        'box-shadow:0 0 0 2px rgba(34,197,94,.3)"></span>'
        f'Online &nbsp;·&nbsp; {n:,} chunks indexed</span></div>'
    )


def _health_offline_html() -> str:
    return (
        '<div style="padding:18px 0;display:flex;justify-content:flex-end;align-items:center">'
        '<span style="display:inline-flex;align-items:center;gap:7px;background:#fff1f2;'
        'border:1px solid #fca5a5;border-radius:20px;padding:5px 14px;font-size:12px;'
        'font-weight:600;color:#b91c1c">'
        '<span style="width:7px;height:7px;background:#ef4444;border-radius:50%"></span>'
        'Offline</span></div>'
    )


def _spinner_html(msg: str) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:9px;color:#7c3aed;font-size:13px;padding:10px 0">'
        '<svg style="animation:spin 1s linear infinite" width="15" height="15" viewBox="0 0 24 24" '
        'fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>'
        f'<style>@keyframes spin{{to{{transform:rotate(360deg)}}}}</style>'
        f'{msg}</div>'
    )


# ── Static HTML fragments ─────────────────────────────────────────────────────

_EMPTY_METRICS = (
    '<p style="color:#d1d5db;font-size:13px;padding:8px 0">'
    'Ask a question to see pipeline metrics.</p>'
)
_EMPTY_CITE = (
    '<p style="color:#d1d5db;font-size:13px;padding:8px 0">'
    'Citation verification will appear here.</p>'
)
_EMPTY_SRC = (
    '<p style="color:#d1d5db;font-size:13px;padding:8px 0">'
    'Retrieved source chunks will appear here.</p>'
)
_ABSTAIN_HTML = (
    '<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;'
    'padding:10px 14px;font-size:13px;color:#92400e;font-weight:600">'
    '⚠️ Abstained — confidence below threshold. No answer generated.</div>'
)
_CLEAR_OUTPUTS = (
    [],
    _EMPTY_METRICS,
    _EMPTY_CITE,
    _EMPTY_SRC,
)


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
footer { display: none !important; }
.gradio-container { padding-top: 0 !important; }
#verity-chat { min-height: 500px !important; }
.tab-nav button { font-weight: 600; font-size: 13px; }
"""


# ── Main builder ───────────────────────────────────────────────────────────────

def build_demo(
    pipeline: Any,
    ingest_gen: Callable[[bytes, str, bool], Iterator[dict]],
) -> gr.Blocks:

    # ── respond ───────────────────────────────────────────────────────────────

    def respond(
        message: str,
        history: list,
        threshold: float,
    ) -> Generator:
        """Yields (chatbot_history, metrics_html, cite_html, src_html)."""
        if not message.strip():
            yield history, _EMPTY_METRICS, _EMPTY_CITE, _EMPTY_SRC
            return

        # Reconstruct pipeline history from chatbot MessageDicts
        pipeline_hist: list[dict] = []
        pending_q: str | None = None
        for msg in (history or []):
            role    = msg.get("role", "")    if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if role == "user":
                pending_q = content
            elif role == "assistant" and pending_q is not None:
                pipeline_hist.append({"question": pending_q, "answer": content})
                pending_q = None

        new_hist = list(history or []) + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "▌"},
        ]
        yield new_hist, _spinner_html("Retrieving relevant sources…"), _EMPTY_CITE, _EMPTY_SRC

        partial = ""
        meta_html = cite_html = src_html = ""

        try:
            for sse in pipeline.query_stream(
                message,
                history=pipeline_hist[-5:] or None,
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
                    new_hist[-1]["content"] = "▌"
                    yield (
                        new_hist,
                        _spinner_html(f"{n} sources retrieved — generating answer…"),
                        _EMPTY_CITE, _EMPTY_SRC,
                    )

                elif t == "token":
                    partial += evt["text"]
                    new_hist[-1]["content"] = partial + "▌"
                    yield new_hist, meta_html or _spinner_html("Generating…"), cite_html, src_html

                elif t == "abstain":
                    new_hist[-1]["content"] = evt.get("answer", "Insufficient evidence found.")
                    meta_html = _fmt_metrics_html(evt)
                    yield new_hist, meta_html, _ABSTAIN_HTML, _EMPTY_SRC
                    return

                elif t == "meta":
                    new_hist[-1]["content"] = evt.get("answer", partial)
                    meta_html = _fmt_metrics_html(evt)
                    cite_html = _fmt_citations_html(evt.get("citations"))
                    src_html  = _fmt_sources_html(evt.get("chunks", []))
                    yield new_hist, meta_html, cite_html, src_html
                    return

                elif t == "error":
                    new_hist[-1]["content"] = f"⚠️ {evt.get('message', 'Unknown error')}"
                    yield new_hist, _EMPTY_METRICS, _EMPTY_CITE, _EMPTY_SRC
                    return

        except Exception as exc:
            new_hist[-1]["content"] = f"⚠️ {exc}"

        new_hist[-1]["content"] = partial or new_hist[-1]["content"].rstrip("▌")
        yield new_hist, meta_html, cite_html, src_html

    # ── upload ────────────────────────────────────────────────────────────────

    def do_upload(file: str | None, clear_existing: bool) -> Generator:
        """file is a filepath string (Gradio 6 gr.File default type='filepath')."""
        if not file:
            yield "⚠️ Please select a file first."
            return

        filepath   = Path(file)
        filename   = filepath.name
        try:
            file_bytes = filepath.read_bytes()
        except Exception as exc:
            yield f"❌ Could not read file: {exc}"
            return

        size_kb = len(file_bytes) / 1024
        yield f"⏳ Uploading **{filename}** ({size_kb:.0f} KB)…"

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
                    yield (
                        f"✅ **Done!** &nbsp; {docs_t} document{'s' if docs_t != 1 else ''}"
                        f" · {chunks_n:,} chunks indexed"
                    )
                    return
                pct = round(docs_p / max(docs_t, 1) * 100)
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                yield f"`{bar}` {pct}% — {docs_p}/{docs_t} docs · {chunks_n:,} chunks"
        except Exception as exc:
            yield f"❌ Ingest failed: {exc}"

    # ── corpus ────────────────────────────────────────────────────────────────

    def load_corpus() -> str:
        try:
            stats   = pipeline.retriever.corpus_stats()
            total   = stats.get("total_chunks", 0)
            uniq    = stats.get("unique_sources", 0)
            bm25    = stats.get("bm25_built", False)
            sources = stats.get("sources", [])
            lines = [
                f"**{total:,}** chunks indexed &nbsp;·&nbsp; **{uniq}** unique sources"
                f"&nbsp;·&nbsp; {'🟢 BM25 ready' if bm25 else '⚪ BM25 builds on first query'}",
            ]
            if sources:
                lines += ["", "| # | Source | Chunks |", "|---|--------|--------|"]
                for i, s in enumerate(sources, 1):
                    lines.append(f"| {i} | `{s['name']}` | {s['chunks']:,} |")
            else:
                lines.append("\n*No sources indexed yet. Upload a document from the **Index** tab.*")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ {exc}"

    def get_health_html() -> str:
        try:
            n = pipeline.retriever.collection_size()
            return _health_html(n)
        except Exception:
            return _health_offline_html()

    # ── Blocks ────────────────────────────────────────────────────────────────

    with gr.Blocks(title="Verity", css=_CSS) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            gr.HTML("""
            <div style="padding:20px 0 14px;display:flex;align-items:center;gap:16px">
              <div style="width:46px;height:46px;flex-shrink:0;
                          background:linear-gradient(135deg,#7c3aed 0%,#4f46e5 100%);
                          border-radius:13px;display:flex;align-items:center;
                          justify-content:center;box-shadow:0 4px 16px rgba(124,58,237,.35)">
                <span style="font-size:22px;line-height:1;filter:brightness(2)">✦</span>
              </div>
              <div>
                <h1 style="margin:0;font-size:28px;font-weight:900;letter-spacing:-1px;
                            background:linear-gradient(135deg,#7c3aed 20%,#4338ca 80%);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text">Verity</h1>
                <p style="margin:3px 0 0;font-size:12px;color:#9ca3af;letter-spacing:.03em">
                  Grounded answers &nbsp;·&nbsp; Cited proof &nbsp;·&nbsp; Confident truth
                </p>
              </div>
            </div>""")
            health_out = gr.HTML(
                value=(
                    '<div style="padding:18px 0;display:flex;justify-content:flex-end;'
                    'align-items:center"><span style="color:#d1d5db;font-size:12px">checking…</span></div>'
                ),
            )

        gr.HTML(
            '<div style="height:1px;background:linear-gradient(90deg,'
            'transparent,rgba(124,58,237,.25) 30%,rgba(79,70,229,.25) 70%,transparent);'
            'margin-bottom:18px"></div>'
        )

        with gr.Tabs():

            # ── Chat ────────────────────────────────────────────────────────
            with gr.Tab("💬  Chat"):

                with gr.Row(equal_height=False):

                    # Sidebar
                    with gr.Column(scale=1, min_width=230):
                        gr.HTML(
                            '<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
                            'letter-spacing:.1em;color:#9ca3af;margin-bottom:10px">Settings</div>'
                        )
                        threshold_slider = gr.Slider(
                            0.0, 1.0, value=0.3, step=0.05,
                            label="Abstention Threshold",
                            info="Lower → answers freely  ·  Higher → abstains more",
                        )
                        gr.HTML("""
                        <div style="margin-top:22px;padding-top:18px;
                                    border-top:1px solid rgba(124,58,237,.1)">
                          <div style="font-size:10px;font-weight:800;text-transform:uppercase;
                                      letter-spacing:.1em;color:#9ca3af;margin-bottom:12px">Pipeline</div>
                          <div style="display:flex;flex-direction:column;gap:7px">
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              Hybrid BM25 + Dense
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              Gemini Cross-Encoder Reranker
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              RRF Fusion (k=60)
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              Confidence Gate + Abstention
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              Citation Verifier
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              Agentic Re-retrieval Loop
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              LRU Query Cache
                            </div>
                            <div style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:8px">
                              <span style="width:5px;height:5px;background:#7c3aed;border-radius:50%;
                                           flex-shrink:0;box-shadow:0 0 0 2px rgba(124,58,237,.2)"></span>
                              Parallel Retrieval
                            </div>
                          </div>
                        </div>""")

                    # Chat area
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            elem_id="verity-chat",
                            show_label=False,
                            autoscroll=True,
                            height=500,
                            placeholder=(
                                '<div style="text-align:center;padding:50px 20px;color:#d1d5db">'
                                '<div style="font-size:40px;margin-bottom:14px;opacity:.5">✦</div>'
                                '<div style="font-size:16px;font-weight:700;color:#c4b5fd;margin-bottom:6px">'
                                'Ask Verity anything</div>'
                                '<div style="font-size:13px;color:#d1d5db">'
                                'Upload a corpus first, then ask questions about it</div>'
                                '</div>'
                            ),
                        )
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Ask a question about your corpus…",
                                show_label=False,
                                scale=6,
                                container=False,
                                autofocus=True,
                                lines=1,
                                max_lines=4,
                            )
                            send_btn = gr.Button(
                                "Send  ↵", variant="primary", scale=1, min_width=90,
                            )
                        clear_btn = gr.Button(
                            "↺  Clear conversation",
                            size="sm",
                            variant="secondary",
                        )

                # Metrics / Citations / Sources
                gr.HTML(
                    '<div style="height:1px;background:rgba(124,58,237,.1);margin:16px 0 12px"></div>'
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.HTML(
                            '<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
                            'letter-spacing:.1em;color:#9ca3af;margin-bottom:10px">Pipeline Metrics</div>'
                        )
                        metrics_out = gr.HTML(value=_EMPTY_METRICS)

                    with gr.Column(scale=3):
                        with gr.Accordion("🔗 Citation Verification", open=False):
                            cite_out = gr.HTML(value=_EMPTY_CITE)

                with gr.Accordion("📄 Retrieved Sources", open=False):
                    src_out = gr.HTML(value=_EMPTY_SRC)

                # Wire
                def _submit(msg, hist, thr):
                    yield from respond(msg, hist, thr)

                send_btn.click(
                    _submit,
                    inputs=[msg_box, chatbot, threshold_slider],
                    outputs=[chatbot, metrics_out, cite_out, src_out],
                )
                msg_box.submit(
                    _submit,
                    inputs=[msg_box, chatbot, threshold_slider],
                    outputs=[chatbot, metrics_out, cite_out, src_out],
                )
                send_btn.click(lambda: "", outputs=[msg_box])
                msg_box.submit(lambda: "", outputs=[msg_box])
                clear_btn.click(
                    lambda: _CLEAR_OUTPUTS,
                    outputs=[chatbot, metrics_out, cite_out, src_out],
                )

            # ── Index ───────────────────────────────────────────────────────
            with gr.Tab("📂  Index"):
                gr.HTML("""
                <div style="padding:18px 0 12px">
                  <h3 style="margin:0 0 5px;font-size:19px;font-weight:800;color:#1f2937">
                    Upload &amp; Index a Corpus
                  </h3>
                  <p style="margin:0;font-size:13px;color:#6b7280;line-height:1.6">
                    Accepted formats: &nbsp;<strong>PDF</strong> &nbsp;·&nbsp;
                    <strong>JSONL</strong> (one JSON record per line, must have a <code>text</code> field) &nbsp;·&nbsp;
                    <strong>JSON</strong> array &nbsp;·&nbsp;
                    <strong>CSV</strong> (must have a <code>text</code> column)
                  </p>
                </div>""")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        file_in = gr.File(
                            label="Drop a file here, or click to browse",
                            file_types=[".pdf", ".json", ".jsonl", ".csv"],
                        )
                        with gr.Row():
                            clear_cb = gr.Checkbox(
                                label="Clear existing index first",
                                info="⚠️ Deletes all currently indexed documents",
                                scale=3,
                            )
                            upload_btn = gr.Button(
                                "⬆  Upload & Index", variant="primary", scale=2,
                            )
                    with gr.Column(scale=2):
                        gr.HTML(
                            '<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
                            'letter-spacing:.1em;color:#9ca3af;margin-bottom:10px">Progress</div>'
                        )
                        upload_status = gr.Markdown(
                            value="*Select a file and click Upload & Index.*",
                        )

                upload_btn.click(
                    do_upload,
                    inputs=[file_in, clear_cb],
                    outputs=[upload_status],
                )

            # ── Corpus ──────────────────────────────────────────────────────
            with gr.Tab("🗂️  Corpus"):
                gr.HTML("""
                <div style="padding:18px 0 12px">
                  <h3 style="margin:0 0 5px;font-size:19px;font-weight:800;color:#1f2937">
                    Indexed Sources
                  </h3>
                  <p style="margin:0;font-size:13px;color:#6b7280">
                    BM25 index is built lazily on the first query.
                    Run a question first if the per-source breakdown is empty.
                  </p>
                </div>""")
                with gr.Row():
                    corpus_btn = gr.Button("↺  Refresh stats", variant="secondary", size="sm")
                corpus_md = gr.Markdown("*Click Refresh to load corpus stats.*")
                corpus_btn.click(load_corpus, outputs=[corpus_md])

        demo.load(get_health_html, outputs=[health_out])

    return demo
