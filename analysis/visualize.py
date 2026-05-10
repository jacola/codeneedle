#!/usr/bin/env python3
"""Generate Plotly comparison dashboards from results/*.json.

Layout: one chart per page, grouped under a per-corpus subfolder.

    analysis/charts/
      index.html                          ← top-level: links per corpus
      <corpus>/
        index.html                        ← corpus dashboard with chart links
        leaderboard.html                  ← chart 1 standalone
        per-function.html                 ← chart 2 standalone
        recall-vs-position.html           ← chart 3 standalone

Each chart sizes itself to the data and reserves enough room for a vertical
legend with up to 20 model entries. Every chart is fully interactive — hover,
zoom, pan, click-to-toggle-trace, double-click-to-isolate.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# This file lives in analysis/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import bench…` works regardless of cwd
PASS_THRESHOLD = 8    # matches bench/scorer.py
LEGEND_ROW_PX = 26    # how much vertical space each legend entry needs

# Stable color palette — assigned once per model so every chart uses the same color.
# Models excluded from the side-by-side diff pages (case-insensitive substring
# match against the run's model name). Charts still show every model.
DIFF_EXCLUDE_MODEL_SUBSTRINGS: tuple[str, ...] = ("glm",)


PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#ff9da6", "#9d755d", "#bab0ac", "#b279a2", "#eeca3b",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@dataclass
class Run:
    path: Path
    model: str
    group_name: str
    data: dict


def _group_name(data: dict) -> str:
    files = data.get("files") or ([data["source"]] if data.get("source") else [])
    if not files:
        return "unknown"
    if len(files) == 1:
        return Path(files[0]).stem
    return "+".join(Path(f).stem for f in files[:3])


def load_runs(results_dir: Path) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        if not data.get("results"):
            continue
        group = _group_name(data)
        groups[group].append(Run(path=p, model=data.get("model", p.stem), group_name=group, data=data))
    return groups


def rescore_runs(runs: list[Run], targets: dict) -> None:
    """Recompute primary_matched / primary_total / hallucinated / bonus_matched
    / passed for every result in-place, using the current scorer rules and the
    saved `response` text. Lets pre-existing JSON dumps benefit from later
    scorer fixes (e.g. blank-line auto-credit) without re-running the model.

    Results whose function isn't resolvable from fixtures, or that errored,
    are left untouched.
    """
    from bench.scorer import score, PASS_THRESHOLD

    for run in runs:
        relax = bool(run.data.get("relax_indent", True))
        for res in run.data.get("results", []):
            if res.get("error"):
                continue
            fn = res.get("function")
            if not fn or fn not in targets:
                continue
            response = res.get("response")
            if response is None:
                continue
            t = targets[fn]
            try:
                sc = score(fn, t.primary_lines, t.bonus_lines, response, relax_indent=relax)
            except Exception as e:
                print(f"  (rescore failed for {run.path.name}::{fn}: {e})", file=sys.stderr)
                continue
            res["primary_matched"] = sc.primary_matched
            res["primary_total"] = sc.primary_total
            res["hallucinated"] = sc.hallucinated
            res["bonus_matched"] = sc.bonus_matched
            res["passed"] = sc.primary_matched >= PASS_THRESHOLD


def resolve_targets(runs: list[Run]):
    """Map function name → FunctionTarget. Re-extracts from source so old
    dumps still work. Looks up files by basename under fixtures/ if the
    original absolute path no longer exists (e.g. after a repo rename).
    """
    from bench.extract import extract as bench_extract

    fixtures_dirs = [REPO_ROOT / "fixtures"]
    name_to_target: dict = {}
    tried: set[Path] = set()
    for run in runs:
        for raw in run.data.get("files") or ([run.data.get("source")] if run.data.get("source") else []):
            if not raw:
                continue
            p = Path(raw)
            if not p.exists():
                for d in fixtures_dirs:
                    alt = d / p.name
                    if alt.exists():
                        p = alt
                        break
                else:
                    continue
            if p in tried:
                continue
            tried.add(p)
            try:
                for t in bench_extract(p):
                    name_to_target.setdefault(t.name, t)
            except Exception as e:
                print(f"  (couldn't re-extract {p.name}: {e})", file=sys.stderr)
    return name_to_target


def resolve_line_positions(runs: list[Run]) -> dict[str, int]:
    """Map function name → start_line (back-compat shim)."""
    return {name: t.start_line for name, t in resolve_targets(runs).items()}


def assign_colors(runs: list[Run]) -> dict[str, str]:
    models = sorted({r.model for r in runs})
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _legend_kwargs() -> dict:
    """Right-side vertical legend, padded box, room for many entries."""
    return dict(
        orientation="v",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#ddd",
        borderwidth=1,
        font=dict(size=12),
    )


def _chart_height(*, content_rows: int, n_legend_entries: int, base: int = 420) -> int:
    """Pick a height tall enough for both the data rows and the legend.

    `content_rows` is the number of bars/lines/etc. shown vertically.
    `n_legend_entries` is the legend item count.
    """
    by_legend = LEGEND_ROW_PX * n_legend_entries + 120
    by_content = base + 20 * max(0, content_rows - 8)
    return max(base, by_legend, by_content)


# --- charts ---------------------------------------------------------------


def leaderboard(runs: list[Run], colors: dict[str, str]):
    """Horizontal bar chart, one trace per run (so each is independently
    toggleable from the legend). Sorted best → worst by primary lines matched.
    """
    import plotly.graph_objects as go

    rows = []
    for r in runs:
        matched = sum(x.get("primary_matched", 0) for x in r.data["results"])
        total = sum(x.get("primary_total", 0) for x in r.data["results"])
        passed = sum(1 for x in r.data["results"] if x.get("passed"))
        queries = len(r.data["results"])
        halluc = sum(x.get("hallucinated", 0) for x in r.data["results"])
        errored = sum(1 for x in r.data["results"] if x.get("error"))
        rows.append({
            "model": r.model, "stem": r.path.stem,
            "matched": matched, "total": total,
            "passed": passed, "queries": queries,
            "halluc": halluc, "errored": errored,
        })
    rows.sort(key=lambda d: d["matched"], reverse=True)

    if not rows:
        return None

    max_total = max(r["total"] for r in rows) or 1

    fig = go.Figure()
    for row in rows:
        annotation = (
            f"{row['matched']}/{row['total']} lines · "
            f"{row['passed']}/{row['queries']} pass · "
            f"{row['halluc']} halluc"
            + (f" · {row['errored']} err" if row['errored'] else "")
        )
        hover = (
            f"<b>{row['model']}</b><br>"
            f"file: {row['stem']}<br>"
            f"matched: {row['matched']} / {row['total']}<br>"
            f"pass: {row['passed']} / {row['queries']}<br>"
            f"hallucinated: {row['halluc']}<br>"
            f"errored: {row['errored']}"
        )
        fig.add_trace(go.Bar(
            x=[row["matched"]],
            y=[row["stem"]],
            orientation="h",
            name=row["model"],
            legendgroup=row["model"],
            text=[annotation],
            textposition="outside",
            marker_color=colors[row["model"]],
            marker_line_color="#fff",
            marker_line_width=1,
            hovertext=[hover],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Leaderboard · total primary lines matched (of possible)",
        xaxis=dict(title="lines matched", range=[0, max_total * 1.4]),
        yaxis=dict(autorange="reversed", automargin=True),
        height=_chart_height(content_rows=len(rows), n_legend_entries=len(rows)),
        margin=dict(l=20, r=40, t=70, b=60),
        legend=_legend_kwargs(),
        bargap=0.25,
    )
    return fig


def per_function_bars(runs: list[Run], colors: dict[str, str]):
    """Grouped bars: one bar per (function × run). Dashed line at pass threshold."""
    import plotly.graph_objects as go

    all_fns: set[str] = set()
    for r in runs:
        for x in r.data["results"]:
            all_fns.add(x["function"])

    def mean_score(fn: str) -> float:
        xs = []
        for r in runs:
            x = next((y for y in r.data["results"] if y["function"] == fn), None)
            if x and x.get("primary_total"):
                xs.append(x["primary_matched"] / x["primary_total"])
        return sum(xs) / len(xs) if xs else 0.0

    fns = sorted(all_fns, key=mean_score, reverse=True)
    if not fns:
        return None

    fig = go.Figure()
    total_max = 20
    for r in runs:
        y = []
        for fn in fns:
            x = next((z for z in r.data["results"] if z["function"] == fn), None)
            if x is None or x.get("error"):
                y.append(None)
            else:
                y.append(x.get("primary_matched", 0))
                total_max = max(total_max, x.get("primary_total", 20))
        fig.add_bar(
            x=fns, y=y,
            name=r.model,
            legendgroup=r.model,
            marker_color=colors[r.model],
            customdata=[r.path.stem] * len(fns),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "model: " + r.model + "<br>"
                "run: %{customdata}<br>"
                "matched: %{y}<extra></extra>"
            ),
        )

    fig.add_hline(
        y=PASS_THRESHOLD, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_THRESHOLD})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Per-function score · bars above the dashed line passed",
        xaxis=dict(title="function (sorted by average difficulty)", tickangle=-40,
                   automargin=True),
        yaxis=dict(title="primary lines matched", range=[0, total_max + 2]),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=_chart_height(content_rows=len(fns), n_legend_entries=len(runs)),
        margin=dict(l=70, r=40, t=70, b=160),
        legend=_legend_kwargs(),
    )
    return fig


def recall_vs_depth(runs: list[Run], colors: dict[str, str], positions: dict[str, int]):
    """Scatter + line: X = function start line in source, Y = % matched."""
    import plotly.graph_objects as go

    fig = go.Figure()
    any_data = False
    max_line = 0
    for r in runs:
        pts = []
        for x in r.data["results"]:
            if x.get("error"):
                continue
            fn = x["function"]
            if fn not in positions:
                continue
            total = x.get("primary_total") or 20
            pct = x.get("primary_matched", 0) / total * 100
            pts.append((positions[fn], pct, fn, x.get("primary_matched", 0), total))
        if not pts:
            continue
        any_data = True
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        max_line = max(max_line, max(xs))
        ys = [p[1] for p in pts]
        hover = [
            f"<b>{p[2]}</b><br>line {p[0]:,}<br>"
            f"{p[3]}/{p[4]} matched ({p[1]:.0f}%)"
            f"<br>model: {r.model}<br>run: {r.path.stem}"
            for p in pts
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=r.model,
            legendgroup=r.model,
            line=dict(color=colors[r.model], width=2),
            marker=dict(size=10, color=colors[r.model], line=dict(color="#fff", width=1)),
            hovertext=hover, hoverinfo="text",
        ))

    if not any_data:
        return None

    fig.add_hline(
        y=PASS_THRESHOLD / 20 * 100, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_THRESHOLD}/20 = {PASS_THRESHOLD/20*100:.0f}%)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Recall vs. position in file · left = near top, right = deep",
        xaxis=dict(title="function start line (deeper in file →)",
                   range=[0, max_line * 1.05]),
        yaxis=dict(title="% primary lines matched", range=[-5, 108]),
        height=_chart_height(content_rows=8, n_legend_entries=len(runs), base=520),
        margin=dict(l=70, r=40, t=70, b=70),
        legend=_legend_kwargs(),
    )
    return fig


# --- HTML assembly --------------------------------------------------------


PAGE_CSS = """
  *{box-sizing:border-box;}
  body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:1.5rem 1.25rem;color:#222;
       background:#fafafa;min-height:100vh;}
  .wrap{max-width:1500px;margin:0 auto;}
  header{font-size:.9rem;color:#666;margin-bottom:.5rem;}
  header a{color:#4c78a8;text-decoration:none;}
  header a:hover{text-decoration:underline;}
  header .corpus{font-weight:600;color:#222;}
  nav{margin:.25rem 0 1.5rem 0;font-size:.95rem;border-bottom:1px solid #e5e5e5;padding-bottom:.5rem;}
  nav a{color:#4c78a8;text-decoration:none;margin-right:1rem;padding:.25rem 0;display:inline-block;}
  nav a.active{color:#222;font-weight:600;border-bottom:2px solid #4c78a8;}
  nav a:hover{text-decoration:underline;}
  h1{margin:.25rem 0;font-size:1.5rem;}
  p.caption{color:#555;margin:.25rem 0 1rem 0;font-size:.95rem;line-height:1.5;}
  .chart{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:.5rem;
         box-shadow:0 1px 3px rgba(0,0,0,.04);overflow-x:auto;}
  ul{padding-left:1.25rem;}
  li{margin:.4rem 0;}
  small{color:#888;}

  /* Diff page */
  .diff-wrap{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:.75rem;
             box-shadow:0 1px 3px rgba(0,0,0,.04);overflow-x:auto;}
  table.diff{border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
             font-size:12.5px;width:100%;}
  table.diff th,table.diff td{border:1px solid #e5e5e5;padding:2px 6px;vertical-align:top;
                              white-space:pre;}
  table.diff th{background:#f0f0f0;font-family:system-ui,sans-serif;font-size:12px;text-align:left;
                position:sticky;top:0;z-index:1;}
  table.diff td.lineno{color:#999;text-align:right;width:3.5em;background:#fafafa;font-size:11px;
                       font-family:ui-monospace,monospace;}
  table.diff td.expected{background:#fff;}
  table.diff td.matched{background:#eaf6ea;}
  table.diff td.missing{background:#fbecec;color:#b94a48;text-align:center;font-style:italic;}
  table.diff td.bonus{background:#e8f0fb;color:#214b86;}
  table.diff td.halluc{background:#fff5cc;color:#7a5b00;}
  table.diff td.error{background:#f5f5f5;color:#888;font-style:italic;text-align:center;}
  .legend{display:flex;flex-wrap:wrap;gap:.4rem .8rem;margin:.5rem 0 1rem;font-size:.82rem;color:#444;}
  .legend span{padding:2px 8px;border-radius:3px;border:1px solid #ddd;
               font-family:ui-monospace,monospace;}
  .fn-list{column-count:3;column-gap:1.5rem;list-style:none;padding:0;}
  .fn-list li{break-inside:avoid;margin:.3rem 0;font-family:ui-monospace,monospace;font-size:.9rem;}
  details.extras{margin:1rem 0;}
  details.extras summary{cursor:pointer;font-weight:600;font-size:.9rem;color:#444;padding:.25rem 0;}
  table.extras{border-collapse:collapse;width:100%;font-family:ui-monospace,monospace;font-size:12.5px;
               margin-top:.5rem;}
  table.extras td{border:1px solid #e5e5e5;padding:2px 6px;white-space:pre;vertical-align:top;}
  table.extras td.model{font-family:system-ui;font-size:11px;color:#555;width:14em;background:#fafafa;}
  table.extras td.matched{background:#eaf6ea;}
  table.extras td.bonus{background:#e8f0fb;color:#214b86;}
  table.extras td.halluc{background:#fff5cc;color:#7a5b00;}
  .fn-meta{color:#666;font-size:.85rem;margin:.25rem 0 .75rem 0;}
  .fn-meta code{background:#f3f3f3;padding:1px 4px;border-radius:3px;}

  /* Per-function summary table at top of diff page */
  table.summary{border-collapse:collapse;margin:.25rem 0 1rem 0;font-size:.9rem;
                background:#fff;border:1px solid #e5e5e5;border-radius:6px;overflow:hidden;}
  table.summary th,table.summary td{padding:5px 10px;border-bottom:1px solid #f0f0f0;
                                    vertical-align:middle;text-align:left;}
  table.summary th{background:#f7f7f7;font-weight:600;font-size:.78rem;
                   text-transform:uppercase;letter-spacing:.04em;color:#555;}
  table.summary tbody tr:last-child td{border-bottom:none;}
  table.summary td.sm-model{font-family:ui-monospace,monospace;font-weight:600;}
  table.summary .sm-stem{font-family:system-ui,sans-serif;font-weight:400;
                         color:#888;font-size:.78rem;margin-top:1px;}
  table.summary td.sm-score{font-family:ui-monospace,monospace;}
  .sm-good{display:inline-block;padding:2px 8px;border-radius:3px;background:#d4edda;color:#155724;
           font-weight:600;font-size:.75rem;}
  .sm-ok{display:inline-block;padding:2px 8px;border-radius:3px;background:#eaf6ea;color:#2a6a2a;
         font-weight:600;font-size:.75rem;}
  .sm-bad{display:inline-block;padding:2px 8px;border-radius:3px;background:#fbecec;color:#a32424;
          font-weight:600;font-size:.75rem;}
  .sm-near{display:inline-block;padding:2px 8px;border-radius:3px;background:#fff1d6;color:#8a5a00;
           font-weight:600;font-size:.75rem;}
  .sm-halluc{display:inline-block;padding:1px 6px;border-radius:3px;background:#fff5cc;
             color:#7a5b00;font-size:.78rem;}
  .sm-bonus{display:inline-block;padding:1px 6px;border-radius:3px;background:#e8f0fb;
            color:#214b86;font-size:.78rem;}
"""


CHART_PAGES = [
    # (slug, title, caption, chart_fn_key)
    ("leaderboard", "Leaderboard",
     "Total primary lines matched across all tested functions, sorted so the top bar is the best run. "
     "Each model has its own legend entry — click to hide/show, double-click to isolate. "
     "`halluc` = lines the model emitted that don't match the expected window.",
     "leaderboard"),
    ("per-function", "Per-function score",
     "One bar per model for each function, sorted left-to-right easiest → hardest. "
     "Bars above the dashed line passed (≥ 8 of 20 primary lines matched). "
     "Toggle a model in the legend to remove it from every cluster.",
     "per_function"),
    ("recall-vs-position", "Recall vs. position in file",
     "Each marker is a function placed at its line number in the source. "
     "If recall falls off as x increases, the model is losing context as depth grows — the "
     "core finding for sliding-window models. Hover any marker for details.",
     "recall_vs_position"),
]


def _build_nav(nav_pages: list[tuple[str, str, str]], active: str) -> str:
    return " ".join(
        f'<a href="{href}" class="{"active" if slug == active else ""}">{html.escape(title)}</a>'
        for slug, title, href in nav_pages
    )


def _html_doc(title: str, body: str) -> str:
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )


def write_chart_page(out_path: Path, group: str, slug: str, title: str, caption: str,
                     fig, nav_pages: list[tuple[str, str, str]]) -> None:
    import plotly.io as pio

    nav_links = _build_nav(nav_pages, slug)
    chart_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )

    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a> · '
        f'<span class="corpus">{html.escape(group)}</span></header>'
        f'<nav>{nav_links}</nav>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p class="caption">{caption}</p>'
        f'<div class="chart">{chart_html}</div>'
        f'</div>'
    )
    out_path.write_text(_html_doc(f"{group} · {title}", body))


# --- diff pages ----------------------------------------------------------


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s) or "_"


def _align_for_diff(target, response: str, relax_indent: bool):
    """Re-run scorer alignment to get an exp_idx → pred_idx mapping plus
    per-line tags for the prediction."""
    from difflib import SequenceMatcher
    from bench.scorer import _clean_output, _norm, _norm_relaxed

    norm = _norm_relaxed if relax_indent else _norm
    primary = target.primary_lines
    bonus = target.bonus_lines
    exp_primary = [norm(l) for l in primary]
    exp_bonus = [norm(l) for l in bonus]
    exp_full = exp_primary + exp_bonus

    pred_raw = _clean_output(response or "")
    pred = [norm(l) for l in pred_raw]
    while pred and pred[-1] == "":
        pred.pop()
    pred_raw = pred_raw[: len(pred)]

    sm = SequenceMatcher(a=exp_full, b=pred, autojunk=False)
    exp_to_pred: dict[int, int] = {}
    pred_kind = [-1] * len(pred)
    for block in sm.get_matching_blocks():
        if block.size == 0:
            continue
        for i in range(block.size):
            ei = block.a + i
            pi = block.b + i
            if ei < len(exp_primary):
                exp_to_pred[ei] = pi
                pred_kind[pi] = 0
            else:
                pred_kind[pi] = 1

    expected_display = [l.rstrip() for l in primary]
    pred_display = [l.rstrip() for l in pred_raw]
    return expected_display, pred_display, exp_to_pred, pred_kind


DIFF_LEGEND = (
    '<div class="legend">'
    '<span class="matched">matched</span>'
    '<span class="missing">missing</span>'
    '<span class="halluc">hallucinated</span>'
    '<span class="bonus">bonus (past primary 20)</span>'
    '</div>'
)


def _render_function_diff(fn: str, runs_for_fn: list[tuple[Run, dict | None]],
                          target) -> str:
    aligned = []
    expected_display = [l.rstrip() for l in target.primary_lines]
    primary_total = len(expected_display)
    for run, res in runs_for_fn:
        if res is None:
            continue
        err = res.get("error")
        if err:
            aligned.append((run, res, expected_display, [], {}, [], err))
            continue
        relax = bool(run.data.get("relax_indent", True))
        ed, pd, e2p, pk = _align_for_diff(target, res.get("response") or "", relax)
        aligned.append((run, res, ed, pd, e2p, pk, None))

    if not aligned:
        return f'<h2 class="fn">{html.escape(fn)}</h2><p class="caption">No runs covered this function.</p>'

    header_cells = [
        '<th style="width:3.5em">#</th>',
        '<th>expected</th>',
    ]
    for run, res, ed, pd, e2p, pk, err in aligned:
        if err:
            score_str = ' &nbsp;<small style="color:#b94a48">ERROR</small>'
        else:
            blank_credit = sum(
                1 for i, line in enumerate(expected_display)
                if line.strip() == "" and i not in e2p
            )
            score_str = f" &nbsp;<small>{len(e2p) + blank_credit}/{primary_total}</small>"
        header_cells.append(
            f'<th><div>{html.escape(run.model)}</div>'
            f'<div style="font-weight:400;color:#777;font-size:11px">{html.escape(run.path.stem)}{score_str}</div></th>'
        )

    rows = []
    for i, exp_line in enumerate(expected_display):
        is_blank = exp_line.strip() == ""
        cells = [
            f'<td class="lineno">{target.start_line + i}</td>',
            f'<td class="expected">{html.escape(exp_line) or "&nbsp;"}</td>',
        ]
        for run, res, ed, pd, e2p, pk, err in aligned:
            if err:
                cells.append('<td class="error">ERROR</td>')
                continue
            if i in e2p:
                pi = e2p[i]
                txt = pd[pi] if pi < len(pd) else ""
                cells.append(f'<td class="matched">{html.escape(txt) or "&nbsp;"}</td>')
            elif is_blank:
                cells.append('<td class="matched" style="color:#bbb;font-style:italic">(blank — auto)</td>')
            else:
                cells.append('<td class="missing">—</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')

    table_html = (
        '<div class="diff-wrap">'
        f'<table class="diff"><thead><tr>{"".join(header_cells)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '</div>'
    )

    # Per-model summary table (sorted worst → best so trouble cases jump out).
    summary_rows_data = []
    for run, res, ed, pd, e2p, pk, err in aligned:
        if err:
            summary_rows_data.append({
                "model": run.model, "stem": run.path.stem,
                "matched": -1, "total": primary_total, "halluc": 0,
                "bonus": 0, "err": err, "perfect": False,
            })
            continue
        blank_credit = sum(
            1 for i, line in enumerate(expected_display)
            if line.strip() == "" and i not in e2p
        )
        matched = len(e2p) + blank_credit
        halluc = sum(
            1 for pi, kind in enumerate(pk)
            if kind == -1 and (pi >= len(pd) or pd[pi].strip() != "")
        )
        bonus = sum(1 for kind in pk if kind == 1)
        summary_rows_data.append({
            "model": run.model, "stem": run.path.stem,
            "matched": matched, "total": primary_total,
            "halluc": halluc, "bonus": bonus, "err": None,
            "perfect": matched == primary_total and halluc == 0,
        })

    summary_rows_data.sort(key=lambda d: (d["err"] is None, d["matched"], -d["halluc"]))

    summary_rows = []
    for d in summary_rows_data:
        if d["err"]:
            badge = '<span class="sm-bad">ERROR</span>'
            score_cell = f'<td class="sm-score" colspan="2" style="color:#888;font-style:italic">{html.escape(d["err"])[:120]}</td>'
            extras_cell = ''
        else:
            pct = (d["matched"] / d["total"] * 100) if d["total"] else 0
            gap = d["total"] - d["matched"]
            if d["perfect"]:
                badge = '<span class="sm-good">perfect</span>'
            elif gap == 0:
                # All expected lines matched, but the model also emitted noise.
                badge = '<span class="sm-good">all matched</span>'
            elif d["matched"] >= 8 and gap <= 2:
                badge = f'<span class="sm-near">off by {gap}</span>'
            elif d["matched"] >= 8:
                badge = '<span class="sm-ok">pass</span>'
            else:
                badge = '<span class="sm-bad">fail</span>'
            score_cell = (
                f'<td class="sm-score">{d["matched"]}/{d["total"]} '
                f'<small style="color:#888">({pct:.0f}%)</small></td>'
            )
            extras_bits = []
            if d["halluc"]:
                extras_bits.append(f'<span class="sm-halluc">{d["halluc"]} halluc</span>')
            if d["bonus"]:
                extras_bits.append(f'<span class="sm-bonus">{d["bonus"]} bonus</span>')
            extras_cell = f'<td class="sm-extras">{" ".join(extras_bits) or "&nbsp;"}</td>'
        summary_rows.append(
            f'<tr><td class="sm-model">{html.escape(d["model"])}'
            f'<div class="sm-stem">{html.escape(d["stem"])}</div></td>'
            f'<td class="sm-badge">{badge}</td>'
            f'{score_cell}{extras_cell}</tr>'
        )

    summary_html = (
        '<table class="summary"><thead><tr>'
        '<th>model</th><th></th><th>score</th><th>extras</th>'
        f'</tr></thead><tbody>{"".join(summary_rows)}</tbody></table>'
    ) if summary_rows else ""

    extras_rows = []
    extras_count = 0
    for run, res, ed, pd, e2p, pk, err in aligned:
        if err:
            continue
        for pi, kind in enumerate(pk):
            if kind == 0:
                continue
            text = pd[pi] if pi < len(pd) else ""
            if kind == -1 and not text.strip():
                continue
            cls = "halluc" if kind == -1 else "bonus"
            label = "halluc" if kind == -1 else "bonus"
            extras_rows.append(
                f'<tr><td class="model">{html.escape(run.model)}<br>'
                f'<span style="color:#999">{label}</span></td>'
                f'<td class="{cls}">{html.escape(text) or "&nbsp;"}</td></tr>'
            )
            extras_count += 1

    extras_html = ""
    if extras_rows:
        extras_html = (
            f'<details class="extras" open><summary>Extras emitted by models '
            f'({extras_count} line(s) outside the expected primary window)</summary>'
            f'<table class="extras"><tbody>{"".join(extras_rows)}</tbody></table>'
            f'</details>'
        )

    return (
        f'<p class="fn-meta">starts at line <code>{target.start_line}</code> · '
        f'{primary_total} primary lines · {len(target.bonus_lines)} bonus lines available</p>'
        f'{summary_html}'
        f'{table_html}{extras_html}'
    )


def write_diff_index(out_path: Path, group: str, runs: list[Run],
                     fn_entries: list[tuple[str, str, bool]],
                     nav_pages: list[tuple[str, str, str]]) -> None:
    nav_links = _build_nav(nav_pages, "diff")
    items = []
    for fn, fname, has_target in fn_entries:
        if has_target:
            items.append(f'<li><a href="{fname}">{html.escape(fn)}</a></li>')
        else:
            items.append(f'<li><span style="color:#999">{html.escape(fn)} <small>(source not found)</small></span></li>')
    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a> · '
        f'<span class="corpus">{html.escape(group)}</span></header>'
        f'<nav>{nav_links}</nav>'
        f'<h1>Side-by-side diff</h1>'
        f'<p class="caption">One page per function. Each row is an expected primary line; '
        f'each column is a model run. Green cells = matched, red “—” = missing. '
        f'Lines a model emitted that don\u2019t land in the expected primary window are '
        f'collected in an <em>Extras</em> section below the table.</p>'
        f'{DIFF_LEGEND}'
        f'<ul class="fn-list">{"".join(items)}</ul>'
        f'</div>'
    )
    out_path.write_text(_html_doc(f"{group} · diff", body))


def write_function_diff_page(out_path: Path, group: str, fn: str,
                             runs_for_fn: list[tuple[Run, dict | None]],
                             target,
                             nav_pages: list[tuple[str, str, str]],
                             prev_fn: tuple[str, str] | None,
                             next_fn: tuple[str, str] | None) -> None:
    nav_links = _build_nav(nav_pages, "diff")

    siblings = []
    if prev_fn:
        siblings.append(f'<a href="{prev_fn[1]}">← {html.escape(prev_fn[0])}</a>')
    siblings.append('<a href="diff.html">function index</a>')
    if next_fn:
        siblings.append(f'<a href="{next_fn[1]}">{html.escape(next_fn[0])} →</a>')
    sibling_nav = ' &nbsp;·&nbsp; '.join(siblings)

    diff_html = _render_function_diff(fn, runs_for_fn, target)
    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a> · '
        f'<span class="corpus">{html.escape(group)}</span></header>'
        f'<nav>{nav_links}</nav>'
        f'<h1>Diff: <code>{html.escape(fn)}</code></h1>'
        f'<p class="caption">{sibling_nav}</p>'
        f'{DIFF_LEGEND}'
        f'{diff_html}'
        f'</div>'
    )
    out_path.write_text(_html_doc(f"{group} · diff · {fn}", body))


def write_corpus_index(out_path: Path, group: str, runs: list[Run],
                       generated_pages: list[tuple[str, str, str]]) -> None:
    models = sorted({r.model for r in runs})
    queries = sum(len(r.data["results"]) for r in runs)
    items = "".join(
        f'<li><a href="{href}">{html.escape(title)}</a></li>'
        for _slug, title, href in generated_pages
    )
    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a></header>'
        f'<h1>{html.escape(group)}</h1>'
        f'<p class="caption">{len(runs)} run(s) · {queries} queries · '
        f'{len(models)} unique model(s): {html.escape(", ".join(models))}</p>'
        f'<ul>{items}</ul>'
        f'</div>'
    )
    out_path.write_text(_html_doc(f"{group} · charts", body))


def write_dashboard(group: str, runs: list[Run], out_dir: Path) -> list[tuple[str, str, str]]:
    """Write all chart pages + diff pages for one corpus.
    Returns nav-style list of (slug, title, href) for the corpus index.
    """
    colors = assign_colors(runs)
    targets = resolve_targets(runs)
    rescore_runs(runs, targets)
    positions = {n: t.start_line for n, t in targets.items()}

    figs = {
        "leaderboard": leaderboard(runs, colors),
        "per_function": per_function_bars(runs, colors),
        "recall_vs_position": recall_vs_depth(runs, colors, positions),
    }

    chart_dir = out_dir / group
    chart_dir.mkdir(parents=True, exist_ok=True)

    # Build the full nav list first so every page gets the same links.
    nav_pages: list[tuple[str, str, str]] = []
    for slug, title, _caption, fig_key in CHART_PAGES:
        if figs.get(fig_key) is not None:
            nav_pages.append((slug, title, f"{slug}.html"))

    fn_set: set[str] = set()
    for r in runs:
        for x in r.data["results"]:
            fn_set.add(x["function"])
    fns = sorted(
        fn_set,
        key=lambda f: (targets[f].start_line if f in targets else 10**9, f),
    )
    # Filter runs for the diff view only (charts still include everything).
    diff_runs = [
        r for r in runs
        if not any(s in r.model.lower() for s in DIFF_EXCLUDE_MODEL_SUBSTRINGS)
    ]
    diff_available = bool(diff_runs) and any(f in targets for f in fns)
    if diff_available:
        nav_pages.append(("diff", "Diff", "diff.html"))

    # Chart pages.
    generated: list[tuple[str, str, str]] = []
    for slug, title, caption, fig_key in CHART_PAGES:
        fig = figs.get(fig_key)
        if fig is None:
            continue
        page_path = chart_dir / f"{slug}.html"
        write_chart_page(page_path, group, slug, title, caption, fig, nav_pages)
        generated.append((slug, title, f"{slug}.html"))

    # Diff pages.
    if diff_available:
        fn_entries: list[tuple[str, str, bool]] = []
        renderable: list[tuple[str, str]] = []
        for fn in fns:
            fname = f"diff__{_slugify(fn)}.html"
            has_target = fn in targets
            fn_entries.append((fn, fname, has_target))
            if has_target:
                renderable.append((fn, fname))

        for idx, (fn, fname) in enumerate(renderable):
            runs_for_fn = []
            for r in diff_runs:
                res = next((x for x in r.data["results"] if x["function"] == fn), None)
                runs_for_fn.append((r, res))
            prev_fn = renderable[idx - 1] if idx > 0 else None
            next_fn = renderable[idx + 1] if idx + 1 < len(renderable) else None
            write_function_diff_page(
                chart_dir / fname, group, fn, runs_for_fn, targets[fn],
                nav_pages, prev_fn, next_fn,
            )

        write_diff_index(chart_dir / "diff.html", group, diff_runs, fn_entries, nav_pages)
        generated.append(("diff", "Diff", "diff.html"))

    write_corpus_index(chart_dir / "index.html", group, runs, generated)
    return generated


def write_top_index(groups: dict[str, list[Run]], out_dir: Path) -> Path:
    idx = out_dir / "index.html"
    items = []
    for name in sorted(groups):
        runs = groups[name]
        models = sorted({r.model for r in runs})
        items.append(
            f'<li><a href="{name}/index.html">{name}</a> '
            f'<small>— {len(runs)} run(s), {len(models)} model(s): {", ".join(models)}</small></li>'
        )
    body = (
        f'<div class="wrap">'
        f'<h1>codeneedle · benchmark dashboards</h1>'
        f'<ul>{"".join(items)}</ul>'
        f'</div>'
    )
    idx.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>codeneedle dashboards</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: analysis/charts/")
    args = ap.parse_args(argv)

    out_dir = args.output_dir or (REPO_ROOT / "analysis" / "charts")
    groups = load_runs(args.results_dir)
    if not groups:
        print(f"no usable result JSON files in {args.results_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(len(r) for r in groups.values())
    print(f"Loaded {total_runs} run(s) in {len(groups)} group(s)")
    for name, runs in sorted(groups.items()):
        generated = write_dashboard(name, runs, out_dir)
        slugs = ", ".join(s for s, _t, _h in generated)
        print(f"  {name}: {len(runs)} run(s) → {out_dir / name}/{{ {slugs} }}.html")

    idx = write_top_index(groups, out_dir)
    print(f"\nopen {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
