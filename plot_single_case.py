#!/usr/bin/env python3
"""Interactive UMAP visualization of 100 reasoning traces for a single case.

This script:
- Reads per-case CSV and selects the case with accuracy closest to 50% (or a provided --pmcid)
- Loads the corresponding runs from the per-run CSV (100 repeats expected)
- Splits each run's reasoning_trace into sentences
- Embeds unique sentences using OpenAI embeddings (256 dims by default)
- Reduces embeddings to 2D via UMAP (optionally preceded by a small PCA for speed)
- Builds an interactive Plotly HTML with a slider 1..N (N = max sentences across runs)
  where each run is drawn as a connected polyline of its sentence embeddings, colored by correctness:
  blue = correct, red = incorrect. For slider position i: show path up to i; older points faded; latest point highlighted.

Usage example:
  OPENAI_API_KEY=... ./plot_single_case_traces.py \
    --per_case_csv results_repeat_per_case.csv \
    --runs_csv results_repeat_runs.csv \
    --output single_case_traces.html

Requires: pandas, numpy, scikit-learn, plotly, openai, tqdm, umap-learn

Reference: UMAP documentation — https://umap-learn.readthedocs.io/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
import json
import html as html_mod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from openai import OpenAI  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
import umap
from tqdm.auto import tqdm  # type: ignore


# --------- Utilities ---------

def simple_sentence_split(text: str) -> List[str]:
    """A lightweight sentence splitter without external models.

    Splits on punctuation (.!?) followed by whitespace OR line boundaries, then strips.
    Keeps non-empty sentences, preserving order.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.replace("\r", " "))
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in parts if s and s.strip()]
    return sentences


def parse_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def rgba(color: Tuple[int, int, int], alpha: float) -> str:
    r, g, b = color
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a})"


def batch(iterable: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield list(iterable[i : i + size])


def embed_sentences(
    client: OpenAI,
    sentences: List[str],
    model: str,
    embedding_dim: Optional[int] = 256,
    batch_size: int = 100,
    retry: int = 3,
    sleep: float = 1.0,
) -> Dict[str, List[float]]:
    unique = list(dict.fromkeys([s.strip() for s in sentences if s and s.strip()]))
    mapping: Dict[str, List[float]] = {}

    use_dimensions = embedding_dim is not None and embedding_dim > 0
    warned_no_dimensions = False

    for chunk in tqdm(list(batch(unique, batch_size)), desc="Embedding", unit="batch"):
        for attempt in range(1, retry + 1):
            try:
                if use_dimensions:
                    resp = client.embeddings.create(model=model, input=chunk, dimensions=embedding_dim)
                else:
                    resp = client.embeddings.create(model=model, input=chunk)
                for s, emb in zip(chunk, resp.data):
                    mapping[s] = emb.embedding  # type: ignore[attr-defined]
                break
            except Exception as exc:  # pylint: disable=broad-exception-caught
                msg = str(exc).lower()
                if use_dimensions and ("dimension" in msg or "invalid" in msg or "unknown parameter" in msg):
                    if not warned_no_dimensions:
                        print(
                            "Note: 'dimensions' not supported for this model/API version; falling back to default embedding size.",
                            file=sys.stderr,
                        )
                        warned_no_dimensions = True
                    use_dimensions = False
                    continue
                if attempt == retry:
                    raise
                time.sleep(sleep * attempt)
    return mapping


def reduce_to_2d(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    n, d = vectors.shape
    if n == 0:
        return np.empty((0, 2))
    if n <= 3:
        pca = PCA(n_components=min(2, d))
        out = pca.fit_transform(vectors)
        if out.shape[1] == 1:
            out = np.hstack([out, np.zeros((n, 1))])
        return out[:, :2]

    # For medium/large n, use UMAP for 2D projection. UMAP works well with cosine distance on text embeddings.
    # See UMAP docs: https://umap-learn.readthedocs.io/
    # Optional speed-up: PCA to 50 dims if very high-dimensional vectors
    X = vectors
    if d > 50:
        X = PCA(n_components=50, random_state=random_state).fit_transform(vectors)

    n_neighbors = int(max(5, min(30, np.sqrt(n))))  # heuristic scaling with dataset size
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        init="spectral",
    )
    return reducer.fit_transform(X)


def wrap_for_hover(text: str, width: int = 70) -> str:
    """Insert <br> breaks to wrap long text for Plotly hover labels."""
    if not text:
        return ""
    words = text.split()
    lines: List[str] = []
    line = ""
    for w in words:
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= width:
            line += " " + w
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "<br>".join(lines)


# --------- Data structures ---------

@dataclass
class RunSentenceItem:
    run_id: str  # e.g., repeat_index as string
    correct: bool
    sent_index: int  # 1-based index within the run
    sentence: str


def collect_items_for_case(runs_df: pd.DataFrame, pmcid: str) -> Tuple[List[RunSentenceItem], int]:
    items: List[RunSentenceItem] = []
    max_len = 0
    case_runs = runs_df[runs_df["pmcid"].astype(str) == str(pmcid)].copy()
    if case_runs.empty:
        return items, max_len

    # Sort by repeat_index for reproducibility
    if "repeat_index" in case_runs.columns:
        case_runs = case_runs.sort_values(by=["repeat_index"])  # 0..N-1

    for _, row in case_runs.iterrows():
        reasoning = row.get("reasoning_trace")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        run_id = str(row.get("repeat_index") if row.get("repeat_index") is not None else row.name)
        correct = parse_bool(row.get("is_correct"))
        sents = simple_sentence_split(reasoning)
        for j, s in enumerate(sents, start=1):
            items.append(RunSentenceItem(run_id=run_id, correct=correct, sent_index=j, sentence=s))
        if len(sents) > max_len:
            max_len = len(sents)

    return items, max_len


def build_plot(
    items: List[RunSentenceItem],
    coords: np.ndarray,
    output_html: str,
    title_suffix: str = "",
    case_prompt: str = "",
    true_diagnosis: str = "",
) -> None:
    # Group indices by run
    by_run: Dict[str, List[int]] = {}
    for idx, it in enumerate(items):
        by_run.setdefault(it.run_id, []).append(idx)
    for rid in by_run:
        by_run[rid].sort(key=lambda i: items[i].sent_index)

    COLOR_CORRECT = (33, 150, 243)   # blue 500
    COLOR_INCORRECT = (244, 67, 54)  # red 500

    # Keep a stable order of runs to avoid any trace reordering across frames
    run_ids = list(by_run.keys())

    # Prepare mapping for side panel: run_id -> list of sentences in order
    run_to_sentences: Dict[str, List[str]] = {}
    for rid, idxs in by_run.items():
        run_to_sentences[rid] = [items[i].sentence for i in idxs]

    # Determine max sentence index across runs for slider
    max_i = max((len(idxs) for idxs in by_run.values()), default=0)

    # Helper to produce masked arrays of fixed length per run so animations don't "jump"
    # Additionally, compute line segments for faded past path and the highlighted latest segment.
    def masked_series(rid: str, upto: int):
        idxs = by_run[rid]
        L = len(idxs)
        x = np.full(L, np.nan, dtype=float)
        y = np.full(L, np.nan, dtype=float)
        # Only reveal first `upto` points; others remain NaN (hidden)
        k = min(upto, L)
        if k > 0:
            coords_idx = np.array(idxs[:k], dtype=int)
            x[:k] = coords[coords_idx, 0]
            y[:k] = coords[coords_idx, 1]
        # Per-point styling: fade older, highlight current, hide future
        col = COLOR_CORRECT if items[idxs[0]].correct else COLOR_INCORRECT
        marker_colors = []
        marker_sizes = []
        texts = []
        custom = []
        for j in range(L):
            sent_idx = items[idxs[j]].sent_index
            if j < k - 1:
                marker_colors.append(rgba(col, 0.01))
                marker_sizes.append(7)
            elif j == k - 1:
                marker_colors.append(rgba(col, 1.0))
                marker_sizes.append(10)
            else:
                marker_colors.append(rgba(col, 0.0))
                marker_sizes.append(6)
            texts.append(
                f"Run: {rid}<br>Idx: {sent_idx}<br>Correct: {items[idxs[0]].correct}<br>Sentence: {wrap_for_hover(items[idxs[j]].sentence)}"
            )
            custom.append([rid, sent_idx])
        # Build past path (faded) and current segment (bright) line coordinates
        # Past path includes points 1..k with low alpha; current segment is only between k-1 and k with high alpha
        past_x = x.copy()
        past_y = y.copy()
        if k <= 1:
            curr_seg_x = np.array([np.nan, np.nan], dtype=float)
            curr_seg_y = np.array([np.nan, np.nan], dtype=float)
        else:
            i_prev = idxs[k - 2]
            i_curr = idxs[k - 1]
            curr_seg_x = np.array([coords[i_prev, 0], coords[i_curr, 0]], dtype=float)
            curr_seg_y = np.array([coords[i_prev, 1], coords[i_curr, 1]], dtype=float)

        return x, y, marker_colors, marker_sizes, texts, custom, past_x, past_y, curr_seg_x, curr_seg_y

    # Build frames: one stable trace per run (lines+markers), with future points masked as NaN
    frames = []
    for i in range(1, max_i + 1):
        frame_data = []
        for rid in run_ids:
            x, y, mcols, msizes, texts, custom, past_x, past_y, curr_seg_x, curr_seg_y = masked_series(rid, i)
            col = COLOR_CORRECT if items[by_run[rid][0]].correct else COLOR_INCORRECT
            # 1) Faded past path line (covers all revealed points)
            frame_data.append(
                go.Scatter(
                    x=past_x,
                    y=past_y,
                    mode="lines",
                    line=dict(color=rgba(col, 0.005), width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # 2) Bright current segment (last two points only)
            frame_data.append(
                go.Scatter(
                    x=curr_seg_x,
                    y=curr_seg_y,
                    mode="lines",
                    line=dict(color=rgba(col, 0.95), width=2.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # 3) Markers with per-point fading/highlight
            frame_data.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=mcols, size=msizes, line=dict(width=0)),
                    customdata=custom,
                    text=texts,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
        frames.append(go.Frame(data=frame_data, name=f"frame_{i}"))

    steps = [
        {
            "args": [[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": str(i),
            "method": "animate",
        }
        for i in range(1, max_i + 1)
    ]

    # Initial data: same structure as frames (three traces per run: past line, current segment, markers)
    init_data = []
    for rid in run_ids:
        x, y, mcols, msizes, texts, custom, past_x, past_y, curr_seg_x, curr_seg_y = masked_series(rid, 1)
        col = COLOR_CORRECT if items[by_run[rid][0]].correct else COLOR_INCORRECT
        # past path (may be a single point -> no visible line yet)
        init_data.append(
            go.Scatter(
                x=past_x,
                y=past_y,
                mode="lines",
                line=dict(color=rgba(col, 0.01), width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # current segment (none at i=1)
        init_data.append(
            go.Scatter(
                x=curr_seg_x,
                y=curr_seg_y,
                mode="lines",
                line=dict(color=rgba(col, 0.95), width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # markers
        init_data.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(color=mcols, size=msizes, line=dict(width=0)),
                customdata=custom,
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    title = "Single Case: 100 Reasoning Traces (UMAP)"
    if title_suffix:
        title += f" — {title_suffix}"

    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        title=title,
        xaxis=dict(title="UMAP 1", zeroline=False, showgrid=True),
        yaxis=dict(title="UMAP 2", zeroline=False, showgrid=True),
        width=900,
        height=850,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 1.05,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 100}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "y": -0.07,
                "x": 0.05,
                "len": 0.9,
                "xanchor": "left",
                "yanchor": "top",
                "steps": steps,
                "currentvalue": {"prefix": "Sentence i = ", "visible": True},
                "pad": {"b": 10, "t": 30},
            }
        ],
    )

    # Write custom HTML with sidebar for case prompt/diagnosis and dynamic trace view
    import plotly.io as pio  # type: ignore

    plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    sidebar_case_prompt = html_mod.escape(case_prompt or "(case prompt unavailable)")
    sidebar_diag = html_mod.escape(true_diagnosis or "(unknown)")
    runs_json = json.dumps(run_to_sentences)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\"/>
    <title>Single Case: 100 Reasoning Traces</title>
    <style>
        body {{ margin: 10px; }}
        .trace-sentence {{ margin: 2px 0; }}
        .trace-sentence.faded {{ opacity: 0.35; }}
        .trace-sentence.highlight {{ background: rgba(255, 235, 59, 0.4); }}
        #sidebar h3 {{ margin: 10px 0 6px 0; font-family: Arial, sans-serif; }}
        #case-prompt, #trace-content {{
            white-space: pre-wrap; line-height: 1.4; overflow-y: auto; border: 1px solid #eee; padding: 8px; border-radius: 6px;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.02);
        }}
    </style>
    <script>
        function escapeHtml(str) {{
            return String(str).replace(/[&<>]/g, function(c) {{ return {{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c] || c; }});
        }}
        function renderTrace(runId, highlightIdx) {{
            const runs = JSON.parse(document.getElementById('runs-json').textContent);
            const sents = runs[runId] || [];
            const cont = document.getElementById('trace-content');
            let html = '';
            for (let i=0; i<sents.length; i++) {{
                const cls = (i+1 === Number(highlightIdx)) ? 'highlight' : ((i+1 > Number(highlightIdx)) ? 'faded' : '');
                html += '<div class="trace-sentence '+cls+'">'+escapeHtml(sents[i])+'</div>';
            }}
            cont.innerHTML = html;
        }}
        function attachHoverHandler() {{
            const plotEl = document.querySelector('#plot-container .plotly-graph-div');
            if (!plotEl) return;
            plotEl.on('plotly_hover', function(data) {{
                if (!data || !data.points || !data.points.length) return;
                const p = data.points[0];
                const cd = (p && p.customdata) || [];
                const runId = cd[0];
                const sentIdx = cd[1];
                if (runId && sentIdx) {{
                    renderTrace(String(runId), Number(sentIdx));
                }}
            }});
        }}
        window.addEventListener('DOMContentLoaded', attachHoverHandler);
    </script>
</head>
<body>
    <div id=\"container\" style=\"display:flex; gap:16px; align-items:flex-start;\">
        <div id=\"sidebar\" style=\"width:34%; max-width:520px; min-width:320px; font-family: Arial, sans-serif;\">
            <h3>Case</h3>
            <div style=\"margin:6px 0;\"><b>Diagnosis:</b> <span id=\"diag-text\">{sidebar_diag}</span></div>
            <div id=\"case-prompt\" style=\"max-height:240px;\">{sidebar_case_prompt}</div>
            <h3 style=\"margin-top:14px;\">Run reasoning</h3>
            <div id=\"trace-content\" style=\"max-height:460px; font-size:13px; color:#333;\"></div>
            <div style=\"font-size:12px;color:#777;margin-top:6px;\">Hover a point to view and highlight the sentence. Later sentences are faded.</div>
        </div>
        <div id=\"plot-container\" style=\"flex:1; min-width:600px;\">{plot_html}</div>
    </div>
    <script type=\"application/json\" id=\"runs-json\">{runs_json}</script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Wrote interactive plot to: {output_html}")


def pick_case_closest_to_half(per_case_df: pd.DataFrame) -> Tuple[str, float, int, int]:
    if "accuracy" not in per_case_df.columns:
        raise ValueError("per-case CSV missing 'accuracy' column")
    if "pmcid" not in per_case_df.columns:
        raise ValueError("per-case CSV missing 'pmcid' column")

    # Ensure numeric accuracy
    def _to_float(x: object) -> float:
        try:
            return float(str(x))
        except Exception:
            return 0.0

    df = per_case_df.copy()
    df["accuracy_val"] = df["accuracy"].apply(_to_float)

    # Prefer cases with more repeats if tie
    if "n_repeats" in df.columns:
        df = df.sort_values(by=["accuracy_val", "n_repeats"], ascending=[True, False], key=lambda s: (s - 0.5).abs())
        # The 'key' arg applies to the first sort key only; we sort twice for clarity
        df["acc_dist"] = (df["accuracy_val"] - 0.5).abs()
        df = df.sort_values(by=["acc_dist", "n_repeats"], ascending=[True, False])
    else:
        df["acc_dist"] = (df["accuracy_val"] - 0.5).abs()
        df = df.sort_values(by=["acc_dist"], ascending=[True])

    row = df.iloc[0]
    pmcid = str(row.get("pmcid"))
    acc = float(row.get("accuracy_val", 0.0))
    n_repeats = int(row.get("n_repeats", 0)) if "n_repeats" in df.columns else 0
    n_correct = int(row.get("n_correct", 0)) if "n_correct" in df.columns else int(round(acc * n_repeats))
    return pmcid, acc, n_repeats, n_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 100 reasoning traces for a single case with UMAP")
    parser.add_argument("--runs_csv", default="MedCaseReasoning DeepSeek 100 traces 100 cases.csv", help="Per-run CSV from eval_repeat.py")
    parser.add_argument("--output", default="single_case_reasoning_traces.html", help="Output HTML file path")
    parser.add_argument("--pmcid", default=None, help="Optional: explicit case id to plot; if omitted, pick closest to 0.5 accuracy")
    parser.add_argument("--embedding_model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Target embedding dimension via 'dimensions' param (set 0 to use model default)")
    parser.add_argument("--batch_size", type=int, default=100, help="Embedding batch size")
    parser.add_argument(
        "--embed_cumulative",
        action="store_true",
        help="If set, embed the cumulative trace up to each sentence instead of the sentence alone",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dimensionality reduction")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.runs_csv):
        print(f"ERROR: Runs CSV not found: {args.runs_csv}", file=sys.stderr)
        sys.exit(1)

    runs_df = pd.read_csv(args.runs_csv)

    target_pmcid = str(args.pmcid)
    # Optional: basic validation
    if not (runs_df["pmcid"].astype(str) == target_pmcid).any():
        print(f"ERROR: No runs found for pmcid '{target_pmcid}' in {args.runs_csv}", file=sys.stderr)
        sys.exit(1)
    acc = None
    n_repeats = runs_df[runs_df["pmcid"].astype(str) == target_pmcid]["repeat_index"].nunique()
    n_correct = int(
        runs_df[runs_df["pmcid"].astype(str) == target_pmcid]["is_correct"].apply(parse_bool).sum()
    )


    acc_str = "unknown" if acc is None else f"{acc:.4f}"
    print(f"Selected pmcid={target_pmcid} with accuracy={acc_str}, repeats={n_repeats}, correct={n_correct}")

    # Collect metadata for sidebar: case prompt & diagnosis
    case_prompt_val = ""
    true_diag_val = ""
    if "case_prompt" in runs_df.columns:
        r = runs_df[runs_df["pmcid"].astype(str) == str(target_pmcid)]
        if not r.empty:
            case_prompt_val = str(r.iloc[0].get("case_prompt") or "")
            true_diag_val = str(r.iloc[0].get("true_diagnosis") or r.iloc[0].get("final_diagnosis") or "")
    if not case_prompt_val:
        # Fallback: try runs_df
        r = runs_df[runs_df["pmcid"].astype(str) == str(target_pmcid)]
        if not r.empty:
            case_prompt_val = str(r.iloc[0].get("case_prompt") or "")
            true_diag_val = str(r.iloc[0].get("true_diagnosis") or r.iloc[0].get("final_diagnosis") or "")

    # Collect per-run sentence items for this case
    items, max_len = collect_items_for_case(runs_df, target_pmcid)
    if not items:
        print("No sentences found to visualize for the selected case.", file=sys.stderr)
        sys.exit(1)
    run_count = len({it.run_id for it in items})
    print(f"Collected {len(items)} sentence instances across {run_count} runs. Max run length = {max_len}.")

    # Build texts to embed (sentence-only or cumulative up to each sentence)
    if args.embed_cumulative:
        # Map (run_id, sent_index) -> cumulative text up to that index
        from collections import defaultdict
        per_run: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for it in items:
            per_run[it.run_id].append((it.sent_index, it.sentence))
        for rid in per_run:
            per_run[rid].sort(key=lambda t: t[0])
        cumul_text_map: Dict[Tuple[str, int], str] = {}
        for rid, seq in per_run.items():
            parts: List[str] = []
            last_idx = 0
            for idx, s in seq:
                # Fill gaps if any by reusing previous parts (defensive)
                if idx != last_idx:
                    # append current sentence and update last_idx
                    parts.append(s)
                    last_idx = idx
                else:
                    # same index shouldn't happen; still append for robustness
                    parts.append(s)
                cumul_text_map[(rid, idx)] = " ".join(parts)
        embed_texts = [cumul_text_map[(it.run_id, it.sent_index)] for it in items]
    else:
        embed_texts = [it.sentence for it in items]

    # Deduplicate texts for embedding
    all_sentences = embed_texts
    client = OpenAI(api_key=api_key)
    sentence_to_emb = embed_sentences(
        client,
        all_sentences,
        model=args.embedding_model,
        embedding_dim=args.embedding_dim if args.embedding_dim > 0 else None,
        batch_size=args.batch_size,
    )

    # Build vectors aligned to items; drop any missing embeddings (should be rare)
    vectors: List[List[float]] = []
    filtered_items: List[RunSentenceItem] = []
    missing = 0
    for it, text in zip(items, embed_texts):
        emb = sentence_to_emb.get(text)
        if emb is None:
            missing += 1
            continue
        filtered_items.append(it)
        vectors.append(emb)
    if missing:
        print(f"Warning: {missing} sentences missing embeddings (skipped).", file=sys.stderr)
        items = filtered_items

    vectors_np = np.array(vectors, dtype=np.float32)
    coords = reduce_to_2d(vectors_np, random_state=args.seed)

    mode_suffix = "cum" if args.embed_cumulative else "sent"
    title_suffix = f"pmcid={target_pmcid}, acc={acc_str}, mode={mode_suffix}"
    build_plot(items, coords, args.output, title_suffix=title_suffix, case_prompt=case_prompt_val, true_diagnosis=true_diag_val)


if __name__ == "__main__":
    main()
