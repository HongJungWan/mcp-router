"""Benchmark orchestrator.

For every catalog size (staircase) it builds the routing context once, then for
every labeled query runs each strategy at each k, recording:
  * fractional recall@k = |gold ∩ exposed| / |gold|   (primary routing metric)
  * hit-rate           = all gold exposed (set-cover)  (secondary)
  * task_success       = a DECOUPLED, self-cardinality ReAct agent (Jaccard, NOT
                         told |gold|) picks from only the exposed tools — a weak
                         SECONDARY signal; we report phi(recall_hit, task_success)
                         so the reader can see it is not just recall re-labeled.
plus the gold tool's rank in the full semantic ranking (cliff trace).

The significance test (McNemar, BH-corrected) is computed on recall_hit — the
routing question ("is hybrid's recall better than semantic's?") — NOT on the weak
task_success signal. Aggregation stratifies by difficulty and uses a cluster
(gold-tool) bootstrap. For k < |gold| a full hit is structurally impossible, so
hit-rate is reported per difficulty and fractional recall is the headline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..config import BenchConfig, DEFAULT
from ..determinism import git_sha
from ..catalog import build_catalog, generate_queries
from ..labeling import label_quality_report
from ..models import Query, RouteOutcome
from ..providers.base import get_embedder, get_llm
from ..routing.base import RoutingContext, get_strategy
from ..tracing import Span, Tracer, make_trace_id
from .agent import get_agent
from .metrics import (cluster_bootstrap_ci, mcnemar, benjamini_hochberg,
                      phi_correlation, percentile, mean)


@dataclass
class BenchArtifacts:
    config: dict
    git_sha: str
    embed_model: str
    llm_model_id: str
    vector_backend: str
    cells: List[dict]                       # per (size, strategy, k) metrics
    cells_by_difficulty: List[dict]         # per (size, strategy, k, difficulty)
    mcnemar: List[dict]                     # BH-corrected strategy comparisons (on recall_hit)
    label_report: dict
    n_queries: int
    phi_recall_success: float               # corr(recall_hit, task_success): low => independent
    tracer: Tracer = field(default_factory=Tracer)


def _gold_ranks(ctx: RoutingContext, query: str, gold_ids: List[int]) -> List[int]:
    q = ctx.embedder.embed(query)
    ranking = ctx.tool_index.search(q, k=None)
    pos = {tid: i + 1 for i, (tid, _) in enumerate(ranking)}
    return [pos.get(g, 10 ** 9) for g in gold_ids]


def run_benchmark(cfg: BenchConfig = DEFAULT) -> BenchArtifacts:
    embedder = get_embedder(cfg.embed_provider, dim=cfg.embed_dim)
    llm = get_llm(cfg.llm_provider, embedder=embedder)
    agent = get_agent(llm)                    # same agent wraps mock or claude llm
    queries: List[Query] = generate_queries(cfg)
    tracer = Tracer()
    outcomes: Dict[Tuple[int, str, int], List[RouteOutcome]] = {}

    for size in cfg.catalog_sizes:
        catalog = build_catalog(size)
        ctx = RoutingContext.build(catalog, embedder, backend=cfg.vector_backend)
        by_id = catalog.by_id()
        for q in queries:
            gold = q.gold_tool_ids
            ranks = _gold_ranks(ctx, q.text, gold)
            for strat_name in cfg.strategies:
                strat = get_strategy(strat_name)
                for k in cfg.k_values:
                    exposed = strat(ctx, q.text, k)
                    exposed_set = set(exposed)
                    hit = set(gold).issubset(exposed_set)
                    frac = len(set(gold) & exposed_set) / len(gold)
                    token_cost = sum(by_id[i].token_cost for i in exposed)
                    selected = agent.run(q.text, [by_id[i] for i in exposed])
                    task_success = set(selected) == set(gold)
                    tid = make_trace_id(size, strat_name, k, q.id)
                    outcomes.setdefault((size, strat_name, k), []).append(RouteOutcome(
                        query_id=q.id, strategy=strat_name, k=k, exposed_tool_ids=exposed,
                        exposed_token_cost=token_cost, recall_hit=hit, recall_fraction=frac,
                        difficulty=q.difficulty, cluster=gold[0],
                        selected_tool_ids=selected, task_success=task_success, trace_id=tid))
                    tracer.record(Span(
                        trace_id=tid, query_id=q.id, strategy=strat_name, k=k,
                        catalog_size=size, difficulty=q.difficulty, gold_tool_ids=gold,
                        gold_ranks=ranks, exposed_count=len(exposed), candidate_count=size,
                        recall_hit=hit, task_success=task_success, exposed_token_cost=token_cost))

    cells = _aggregate(outcomes, cfg)
    by_diff = _aggregate_by_difficulty(outcomes, cfg)
    comparisons = _mcnemar_table(outcomes, cfg)

    big = build_catalog(max(cfg.catalog_sizes))
    label_report = label_quality_report(big, queries, llm, cfg)

    # phi correlation between recall_hit and task_success across all outcomes:
    # high |phi| would mean task_success is just recall re-labeled; low => decoupled.
    flat = [o for outs in outcomes.values() for o in outs]
    phi = phi_correlation([1 if o.recall_hit else 0 for o in flat],
                          [1 if o.task_success else 0 for o in flat])

    return BenchArtifacts(
        config=cfg.to_dict(), git_sha=git_sha(), embed_model=embedder.name,
        llm_model_id=llm.model_id, vector_backend=cfg.vector_backend,
        cells=cells, cells_by_difficulty=by_diff, mcnemar=comparisons,
        label_report=label_report, n_queries=len(queries),
        phi_recall_success=phi, tracer=tracer)


def _cell_metrics(outs, cfg, seed_key):
    frac = [o.recall_fraction for o in outs]
    hits = [1.0 if o.recall_hit else 0.0 for o in outs]
    succ = [1.0 if o.task_success else 0.0 for o in outs]
    clusters = [o.cluster for o in outs]
    toks = [o.exposed_token_cost for o in outs]
    return {
        "n": len(outs),
        "recall_at_k": round(mean(frac), 4),                     # fractional (primary)
        "recall_ci": cluster_bootstrap_ci(frac, clusters, cfg.bootstrap_n, seed_key + "|rf"),
        "hit_rate": round(mean(hits), 4),                         # all-gold (secondary)
        "task_success": round(mean(succ), 4),
        "success_ci": cluster_bootstrap_ci(succ, clusters, cfg.bootstrap_n, seed_key + "|ts"),
        "token_mean": round(mean(toks), 1),
        "token_p95": round(percentile(toks, 95), 1),
    }


def _aggregate(outcomes, cfg) -> List[dict]:
    cells = []
    for (size, strat, k), outs in sorted(outcomes.items()):
        cells.append({"catalog_size": size, "strategy": strat, "k": k,
                      **_cell_metrics(outs, cfg, f"{cfg.seed}|{size}|{strat}|{k}")})
    return cells


def _aggregate_by_difficulty(outcomes, cfg) -> List[dict]:
    rows = []
    for (size, strat, k), outs in sorted(outcomes.items()):
        for diff in ("single", "multi", "ambiguous"):
            sub = [o for o in outs if o.difficulty == diff]
            if not sub:
                continue
            rows.append({"catalog_size": size, "strategy": strat, "k": k, "difficulty": diff,
                         **_cell_metrics(sub, cfg, f"{cfg.seed}|{size}|{strat}|{k}|{diff}")})
    return rows


def _mcnemar_table(outcomes, cfg) -> List[dict]:
    # Computed on recall_hit (routing quality), NOT task_success. The
    # hierarchical-vs-hybrid pair is where keyword collisions make the outcome
    # genuinely two-sided (neither strictly dominates).
    pairs = [("semantic_topk", "hybrid"), ("semantic_topk", "hierarchical"),
             ("hierarchical", "hybrid")]
    rows = []
    for size in cfg.catalog_sizes:
        for k in cfg.k_values:
            for a, b in pairs:
                oa, ob = outcomes.get((size, a, k)), outcomes.get((size, b, k))
                if not oa or not ob:
                    continue
                sa = {o.query_id: (1 if o.recall_hit else 0) for o in oa}
                sb = {o.query_id: (1 if o.recall_hit else 0) for o in ob}
                qids = sorted(set(sa) & set(sb))
                res = mcnemar([sa[q] for q in qids], [sb[q] for q in qids])
                rows.append({"catalog_size": size, "k": k, "metric": "recall_hit",
                             "strategy_a": a, "strategy_b": b, **res})
    # Benjamini-Hochberg across the whole family of comparisons.
    qvals = benjamini_hochberg([r["p_value"] for r in rows])
    for r, q in zip(rows, qvals):
        r["q_value"] = q
        r["q_str"] = f"{q:.2e}" if q < 1e-4 else f"{q:.4f}"
    return rows
