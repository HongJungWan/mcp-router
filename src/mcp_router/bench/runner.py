"""Benchmark orchestrator.

For every catalog size (staircase) it builds the routing context once, then for
every labeled query runs each strategy at each k, records the exposed set,
whether recall was hit, and whether a ReAct agent could complete the task using
only the exposed tools. It also records the gold tool's rank in the full
semantic ranking (for the cliff trace) and computes bootstrap CIs + McNemar
significance between strategies.
"""
from __future__ import annotations

import time
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
from .metrics import bootstrap_ci, mcnemar, percentile, mean


@dataclass
class BenchArtifacts:
    config: dict
    git_sha: str
    embed_model: str
    llm_model_id: str
    vector_backend: str
    cells: List[dict]                       # per (size, strategy, k) metrics
    mcnemar: List[dict]                     # strategy comparisons per (size, k)
    label_report: dict
    n_queries: int
    tracer: Tracer = field(default_factory=Tracer)


def _gold_ranks(ctx: RoutingContext, query: str, gold_ids: List[int]) -> List[int]:
    q = ctx.embedder.embed(query)
    ranking = ctx.tool_index.search(q, k=None)     # full ranking (id, score)
    pos = {tid: i + 1 for i, (tid, _) in enumerate(ranking)}
    return [pos.get(g, 10 ** 9) for g in gold_ids]


def run_benchmark(cfg: BenchConfig = DEFAULT) -> BenchArtifacts:
    embedder = get_embedder(cfg.embed_provider, dim=cfg.embed_dim)
    llm = get_llm(cfg.llm_provider, embedder=embedder)
    agent = get_agent("mock" if cfg.llm_provider == "mock" else "langgraph", llm)
    queries: List[Query] = generate_queries(cfg)
    tracer = Tracer()

    # outcomes[(size, strat, k)] = list[RouteOutcome]
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
                    t0 = time.perf_counter()
                    exposed = strat(ctx, q.text, k)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    exposed_set = set(exposed)
                    recall_hit = set(gold).issubset(exposed_set)
                    token_cost = sum(by_id[i].token_cost for i in exposed)
                    exposed_tools = [by_id[i] for i in exposed]
                    selected = agent.run(q.text, exposed_tools, len(gold))
                    task_success = set(selected) == set(gold)
                    tid = make_trace_id(size, strat_name, k, q.id)
                    outcome = RouteOutcome(
                        query_id=q.id, strategy=strat_name, k=k,
                        exposed_tool_ids=exposed, exposed_token_cost=token_cost,
                        recall_hit=recall_hit, latency_ms=latency_ms,
                        selected_tool_ids=selected, task_success=task_success,
                        trace_id=tid,
                    )
                    outcomes.setdefault((size, strat_name, k), []).append(outcome)
                    tracer.record(Span(
                        trace_id=tid, query_id=q.id, strategy=strat_name, k=k,
                        catalog_size=size, difficulty=q.difficulty,
                        gold_tool_ids=gold, gold_ranks=ranks,
                        exposed_count=len(exposed), candidate_count=size,
                        recall_hit=recall_hit, task_success=task_success,
                        exposed_token_cost=token_cost,
                    ))

    cells = _aggregate(outcomes, cfg)
    comparisons = _mcnemar_table(outcomes, cfg)

    # Label-quality report uses the largest catalog (hardest labeling setting).
    big = build_catalog(max(cfg.catalog_sizes))
    label_report = label_quality_report(big, queries, llm, cfg)

    return BenchArtifacts(
        config=cfg.to_dict(), git_sha=git_sha(), embed_model=embedder.name,
        llm_model_id=llm.model_id, vector_backend=cfg.vector_backend,
        cells=cells, mcnemar=comparisons, label_report=label_report,
        n_queries=len(queries), tracer=tracer,
    )


def _aggregate(outcomes, cfg) -> List[dict]:
    seed_key = str(cfg.seed)
    cells = []
    for (size, strat, k), outs in sorted(outcomes.items()):
        hits = [1.0 if o.recall_hit else 0.0 for o in outs]
        succ = [1.0 if o.task_success else 0.0 for o in outs]
        toks = [o.exposed_token_cost for o in outs]
        lat = [o.latency_ms for o in outs]
        cells.append({
            "catalog_size": size, "strategy": strat, "k": k, "n": len(outs),
            "recall_at_k": round(mean(hits), 4),
            "recall_ci": bootstrap_ci(hits, cfg.bootstrap_n, seed_key + f"|r|{size}|{strat}|{k}"),
            "task_success": round(mean(succ), 4),
            "success_ci": bootstrap_ci(succ, cfg.bootstrap_n, seed_key + f"|s|{size}|{strat}|{k}"),
            "token_mean": round(mean(toks), 1),
            "token_p95": round(percentile(toks, 95), 1),
            "latency_p50_ms": round(percentile(lat, 50), 4),
            "latency_p95_ms": round(percentile(lat, 95), 4),
        })
    return cells


def _mcnemar_table(outcomes, cfg) -> List[dict]:
    pairs = [("semantic_topk", "hybrid"), ("semantic_topk", "hierarchical")]
    rows = []
    for size in cfg.catalog_sizes:
        for k in cfg.k_values:
            for a, b in pairs:
                oa = outcomes.get((size, a, k))
                ob = outcomes.get((size, b, k))
                if not oa or not ob:
                    continue
                sa = {o.query_id: (1 if o.task_success else 0) for o in oa}
                sb = {o.query_id: (1 if o.task_success else 0) for o in ob}
                qids = sorted(set(sa) & set(sb))
                res = mcnemar([sa[q] for q in qids], [sb[q] for q in qids])
                # note: res carries discordant counts 'b'/'c'; keep strategy names
                # under distinct keys to avoid clobbering.
                rows.append({"catalog_size": size, "k": k,
                             "strategy_a": a, "strategy_b": b, **res})
    return rows
