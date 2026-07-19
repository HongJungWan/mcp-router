"""External-validity anchor: do REAL MCP tool catalogs have the near-duplicate
crowding the synthetic benchmark assumes?

Embeds the real tool definitions in data/real_mcp_tools.json (247 tools / 22
servers: modelcontextprotocol reference/archived + firecrawl/tavily/playwright/
mongodb) with bge-small-en-v1.5, and compares their pairwise-cosine "crowding"
against the synthetic catalog under the same embedding.

The statistic is the nearest-neighbour cosine per tool: a high value means the
tool has a near-twin that CAN displace it from a small top-k. Caveats it does
NOT capture: it is a tool↔tool geometry metric, whereas recall@k depends on
query↔tool ranking — so a near-duplicate is closer to a necessary than a
sufficient condition for the cliff. The two NN columns are also not strictly
apples-to-apples (synthetic embed_text is templated with filler tokens + numbered
fake protocol suffixes, which inflates bge cosine vs the real tools' natural
language), and the synthetic NN level is partly an output of the crowding knobs
(core_share, kw_collision). Read the real↔synthetic gap as directional, not a
controlled magnitude.

Run: pip install .[local] && python scripts/pairwise_similarity.py
Writes docs/real-tool-similarity.md.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
ROOT = os.path.join(os.path.dirname(__file__), "..")

import numpy as np
from sentence_transformers import SentenceTransformer

from mcp_router.catalog import build_catalog

MODEL = "BAAI/bge-small-en-v1.5"


def nn_stats(sim: np.ndarray) -> dict:
    """Per-row nearest-neighbour cosine (max off-diagonal) distribution."""
    s = sim.copy()
    np.fill_diagonal(s, -1.0)
    nn = s.max(axis=1)
    return {
        "n": int(sim.shape[0]),
        "nn_mean": round(float(nn.mean()), 3),
        "nn_p50": round(float(np.percentile(nn, 50)), 3),
        "nn_p90": round(float(np.percentile(nn, 90)), 3),
        "nn_max": round(float(nn.max()), 3),
        "frac_nn_gt_0.80": round(float((nn > 0.80).mean()), 3),
        "frac_nn_gt_0.90": round(float((nn > 0.90).mean()), 3),
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # em-dashes over Windows cp949
    except Exception:
        pass
    model = SentenceTransformer(MODEL)

    real = json.load(open(os.path.join(ROOT, "data", "real_mcp_tools.json"), encoding="utf-8"))["tools"]
    r_texts = [f"{t['name']}. {t['description']}" for t in real]
    r_servers = [t["server"] for t in real]
    R = model.encode(r_texts, normalize_embeddings=True)
    r_sim = R @ R.T

    # synthetic catalog under the SAME embedder, SIZE-MATCHED to the real corpus
    # so the NN comparison isn't confounded by catalog size.
    syn = build_catalog(len(real)).tools
    S = model.encode([t.embed_text for t in syn], normalize_embeddings=True)
    s_sim = S @ S.T

    real_stats = nn_stats(r_sim)
    syn_stats = nn_stats(s_sim)

    # within-server vs cross-server mean cosine (real)
    n = len(real)
    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if r_servers[i] == r_servers[j] else cross).append(float(r_sim[i, j]))
    within_mean = round(float(np.mean(within)), 3)
    cross_mean = round(float(np.mean(cross)), 3)

    # top near-duplicate pairs (real)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(r_sim[i, j]), real[i]["name"], real[j]["name"], r_servers[i]))
    pairs.sort(reverse=True)
    top = pairs[:12]

    # ---- report ----
    srv_list = sorted(set(r_servers))
    lines = [
        "# Real MCP tools vs. the synthetic catalog — pairwise similarity",
        "",
        f"Embedding model: `{MODEL}`. Real corpus: **{n} tools / {len(srv_list)} servers** "
        f"({', '.join(srv_list)}), harvested from public MCP servers "
        "(modelcontextprotocol reference/archived + firecrawl/tavily/playwright/mongodb). "
        "Synthetic: `build_catalog(300)`.",
        "",
        "**The question:** do real tool catalogs actually contain the near-duplicate "
        "crowding the benchmark assumes? The nearest-neighbour cosine per tool is the "
        "load-bearing number — a high value means the tool has a near-twin that can "
        "displace it from a small top-k.",
        "",
        "| nearest-neighbour cosine | real MCP tools | synthetic catalog |",
        "|---|---|---|",
        f"| mean | {real_stats['nn_mean']} | {syn_stats['nn_mean']} |",
        f"| median | {real_stats['nn_p50']} | {syn_stats['nn_p50']} |",
        f"| p90 | {real_stats['nn_p90']} | {syn_stats['nn_p90']} |",
        f"| max | {real_stats['nn_max']} | {syn_stats['nn_max']} |",
        f"| fraction with NN > 0.80 | {real_stats['frac_nn_gt_0.80']} | {syn_stats['frac_nn_gt_0.80']} |",
        f"| fraction with NN > 0.90 | {real_stats['frac_nn_gt_0.90']} | {syn_stats['frac_nn_gt_0.90']} |",
        "",
        f"Real, within-server mean cosine = **{within_mean}**, cross-server = **{cross_mean}** "
        "(tools from the same server are the near-duplicates, as expected).",
        "",
        "**Top real near-duplicate pairs (bge cosine):**",
        "",
        "| cosine | tool A | tool B | server |",
        "|---|---|---|---|",
    ]
    for sim, a, b, srv in top:
        lines.append(f"| {sim:.3f} | {a} | {b} | {srv} |")
    lines += ["", "## Reading", ""]
    lines.append(
        f"Real MCP tools carry genuine near-duplicates: median nearest-neighbour "
        f"cosine {real_stats['nn_p50']}, with {int(real_stats['frac_nn_gt_0.80']*100)}% of "
        f"tools having a neighbour above 0.80. So the crowding the benchmark studies is "
        f"NOT a synthetic invention — it exists in real catalogs, concentrated within a "
        f"server (within {within_mean} vs cross {cross_mean})."
    )
    lines.append("")
    verdict = ("comparable to" if abs(real_stats["nn_p50"] - syn_stats["nn_p50"]) < 0.05
               else ("milder than" if real_stats["nn_p50"] < syn_stats["nn_p50"] else "harsher than"))
    lines.append(
        f"Versus the size-matched synthetic catalog, the real corpus is **{verdict}** it on median "
        f"NN cosine (real {real_stats['nn_p50']} vs synthetic {syn_stats['nn_p50']}). Read this as "
        "**directional, not a controlled magnitude**: the synthetic NN level is partly an output of "
        "the crowding knobs (core_share, kw_collision), and the synthetic embed_text is templated "
        "(filler + numbered fake-protocol suffixes) which inflates its bge cosine relative to the "
        "real tools' natural-language descriptions — so the two NN columns are not strictly "
        "apples-to-apples."
    )
    lines.append("")
    lines.append(
        f"Two more caveats. (1) NN cosine is a tool↔tool geometry metric; recall@k depends on "
        "query↔tool ranking, so a near-duplicate is closer to a *necessary* than a *sufficient* "
        "condition for the cliff. (2) "
        f"{n} tools / {len(srv_list)} servers is a real but non-exhaustive sample — an anchor, not a "
        "population estimate."
    )

    out = os.path.join(ROOT, "docs", "real-tool-similarity.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
