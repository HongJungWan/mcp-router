"""Render benchmark artifacts to disk: results.csv, report.md, summary.json,
traces.jsonl, and a pure-Python SVG recall-cliff chart (no matplotlib)."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

from .runner import BenchArtifacts

# Anthropic-ish palette (cream bg, coral accents, slate ink).
_BG = "#F0EEE6"
_INK = "#29261B"
_MUTED = "#8C8575"
_SERIES = ["#C6714F", "#D4A27F", "#B5654A", "#6B8F71", "#8A6D9E"]


def _cell(cells: List[dict], size: int, strat: str, k: int) -> dict | None:
    for c in cells:
        if c["catalog_size"] == size and c["strategy"] == strat and c["k"] == k:
            return c
    return None


def write_all(art: BenchArtifacts, outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "csv": os.path.join(outdir, "results.csv"),
        "md": os.path.join(outdir, "report.md"),
        "json": os.path.join(outdir, "summary.json"),
        "traces": os.path.join(outdir, "traces.jsonl"),
        "svg": os.path.join(outdir, "recall_cliff.svg"),
    }
    _write_csv(art, paths["csv"])
    _write_json(art, paths["json"])
    art.tracer.flush(paths["traces"])
    _write_svg(art, paths["svg"])
    _write_md(art, paths["md"], paths)
    return paths


def _write_csv(art: BenchArtifacts, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        cols = ["catalog_size", "strategy", "k", "n", "recall_at_k", "recall_ci",
                "task_success", "success_ci", "token_mean", "token_p95",
                "latency_p50_ms", "latency_p95_ms"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in art.cells:
            row = dict(c)
            row["recall_ci"] = f"{c['recall_ci'][0]}-{c['recall_ci'][1]}"
            row["success_ci"] = f"{c['success_ci'][0]}-{c['success_ci'][1]}"
            w.writerow(row)


def _write_json(art: BenchArtifacts, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "config": art.config, "git_sha": art.git_sha,
            "embed_model": art.embed_model, "llm_model_id": art.llm_model_id,
            "vector_backend": art.vector_backend, "n_queries": art.n_queries,
            "cells": art.cells, "mcnemar": art.mcnemar,
            "label_report": art.label_report,
            "n_cliff_events": sum(1 for s in art.tracer.spans if s.is_cliff),
        }, f, ensure_ascii=False, indent=2)


def _svg_line_chart(series: List[Tuple[str, str, List[Tuple[float, float]]]],
                    xticks: List[int], title: str) -> str:
    W, H = 840, 470
    ml, mr, mt, mb = 70, 200, 56, 56
    pw, ph = W - ml - mr, H - mt - mb
    xs = sorted(xticks)

    def px(k: int) -> float:
        i = xs.index(k)
        return ml + (pw * i / (len(xs) - 1)) if len(xs) > 1 else ml + pw / 2

    def py(v: float) -> float:
        return mt + ph * (1 - v)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Pretendard, sans-serif">']
    out.append(f'<rect width="{W}" height="{H}" fill="{_BG}"/>')
    out.append(f'<text x="{ml}" y="30" font-size="18" font-weight="700" fill="{_INK}">{title}</text>')
    # y gridlines + labels
    for g in range(6):
        v = g / 5
        y = py(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#DED7C9"/>')
        out.append(f'<text x="{ml-10}" y="{y+4:.1f}" font-size="12" text-anchor="end" fill="{_MUTED}">{v:.1f}</text>')
    # x labels
    for k in xs:
        out.append(f'<text x="{px(k):.1f}" y="{mt+ph+22}" font-size="12" text-anchor="middle" fill="{_MUTED}">k={k}</text>')
    out.append(f'<text x="{ml+pw/2:.0f}" y="{H-12}" font-size="13" text-anchor="middle" fill="{_INK}">exposed top-k</text>')
    out.append(f'<text x="18" y="{mt+ph/2:.0f}" font-size="13" text-anchor="middle" fill="{_INK}" transform="rotate(-90 18 {mt+ph/2:.0f})">recall@k</text>')
    # series
    for si, (label, color, pts) in enumerate(series):
        path = " ".join(f"{'M' if i==0 else 'L'}{px(int(k)):.1f},{py(v):.1f}" for i, (k, v) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for k, v in pts:
            out.append(f'<circle cx="{px(int(k)):.1f}" cy="{py(v):.1f}" r="3.5" fill="{color}"/>')
        ly = mt + 8 + si * 22
        out.append(f'<line x1="{ml+pw+16}" y1="{ly}" x2="{ml+pw+40}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text x="{ml+pw+46}" y="{ly+4}" font-size="12" fill="{_INK}">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _write_svg(art: BenchArtifacts, path: str) -> None:
    ks = art.config["k_values"]
    sizes = art.config["catalog_sizes"]
    big = max(sizes)
    series = []
    # the cliff: semantic_topk recall degrading as catalog grows
    for i, size in enumerate(sizes):
        pts = [(k, (_cell(art.cells, size, "semantic_topk", k) or {}).get("recall_at_k", 0.0)) for k in ks]
        series.append((f"semantic_topk @ {size}", _SERIES[i % len(_SERIES)], pts))
    # recovery at the largest catalog
    for strat, color in (("hierarchical", _SERIES[3]), ("hybrid", _SERIES[4])):
        pts = [(k, (_cell(art.cells, big, strat, k) or {}).get("recall_at_k", 0.0)) for k in ks]
        series.append((f"{strat} @ {big}", color, pts))
    svg = _svg_line_chart(series, ks, "Recall@k — semantic-topk cliff vs recovery")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def _md_table(art: BenchArtifacts, size: int) -> str:
    ks = art.config["k_values"]
    strats = art.config["strategies"]
    head = "| strategy | " + " | ".join(f"recall@{k}" for k in ks) + " | tokens(p95) |"
    sep = "|" + "---|" * (len(ks) + 2)
    lines = [head, sep]
    for s in strats:
        cells = [_cell(art.cells, size, s, k) for k in ks]
        rec = " | ".join(f"{(c or {}).get('recall_at_k', 0):.2f}" for c in cells)
        tok = (_cell(art.cells, size, s, ks[-1]) or {}).get("token_p95", 0)
        lines.append(f"| {s} | {rec} | {tok:.0f} |")
    return "\n".join(lines)


def _write_md(art: BenchArtifacts, path: str, paths: Dict[str, str]) -> None:
    L = art.label_report
    n_cliff = sum(1 for s in art.tracer.spans if s.is_cliff)
    lines = [
        "# MCP Gateway — tool-routing benchmark (M3)",
        "",
        "**Reproducibility envelope**",
        "",
        f"- git_sha: `{art.git_sha}`  · seed: `{art.config['seed']}`  · embed: `{art.embed_model}`  · llm: `{art.llm_model_id}`  · vector: `{art.vector_backend}`",
        f"- queries: {art.n_queries}  · catalog staircase: {art.config['catalog_sizes']}  · k sweep: {art.config['k_values']}",
        f"- bootstrap resamples: {art.config['bootstrap_n']}",
        "",
        "## Label quality (human-verified subset)",
        "",
        f"- human-verified queries: {L['human_verified_n']}  · labeled pairs: {L['labeled_pairs']}  · disagreements: {L['disagreements']}",
        f"- **Cohen's κ = {L['cohen_kappa']}**  · raw agreement = {L['raw_agreement']}  · labeler = `{L['labeler_model']}`",
        "",
    ]
    for size in art.config["catalog_sizes"]:
        lines += [f"## recall@k — catalog size {size}", "", _md_table(art, size), ""]
    lines += ["## Recall cliff", "",
              f"Detected **{n_cliff}** cliff events (gold tool exists but was ranked "
              f"past the exposed top-k under semantic routing). See `traces.jsonl`.",
              "", "Example cliff traces:", ""]
    for s in art.tracer.cliff_events(limit=6):
        lines.append(f"- q#{s.query_id} [{s.difficulty}] {s.strategy}@{s.catalog_size} k={s.k}: "
                     f"gold ranks {s.gold_ranks} (dropped past k) · task_success={s.task_success} · trace `{s.trace_id}`")
    lines += ["", "## McNemar (task success, paired by query)", "",
              "A beats B on `c` queries where B fails and A succeeds; `b` is the reverse.",
              "", "| size | k | A | B | b | c | χ² | p |", "|---|---|---|---|---|---|---|---|"]
    for m in art.mcnemar:
        if m["k"] in (1, 3):
            lines.append(f"| {m['catalog_size']} | {m['k']} | {m['strategy_a']} | {m['strategy_b']} | "
                         f"{m['b']} | {m['c']} | {m['chi2']} | {m['p_value']} |")
    lines += ["", f"![recall cliff]({os.path.basename(paths['svg'])})", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
