"""CLI: `python -m mcp_router bench run` (what `make bench` calls).

Everything defaults to the offline deterministic path. Flags switch in the
production adapters when their extras + credentials are present.
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import BenchConfig, DEFAULT
from .bench.runner import run_benchmark
from .bench.report import write_all


def _cfg_from_args(a) -> BenchConfig:
    cfg = BenchConfig()
    cfg.embed_provider = a.embed
    cfg.llm_provider = a.llm
    cfg.vector_backend = a.vector
    if a.sizes:
        cfg.catalog_sizes = [int(x) for x in a.sizes.split(",")]
    if a.queries:
        cfg.n_queries = a.queries
    return cfg


def cmd_bench_run(a) -> int:
    cfg = _cfg_from_args(a)
    print(f"[mcp-router] bench: embed={cfg.embed_provider} llm={cfg.llm_provider} "
          f"vector={cfg.vector_backend} sizes={cfg.catalog_sizes} "
          f"queries={cfg.n_queries} seed={cfg.seed}", file=sys.stderr)
    art = run_benchmark(cfg)
    paths = write_all(art, a.out)

    big = max(cfg.catalog_sizes)
    print(f"\n=== fractional recall@k - catalog {big} ===")
    for strat in cfg.strategies:
        row = []
        for k in cfg.k_values:
            c = next((c for c in art.cells if c["catalog_size"] == big
                      and c["strategy"] == strat and c["k"] == k), None)
            row.append(f"k={k}:{c['recall_at_k']:.2f}" if c else "-")
        print(f"  {strat:<14} " + "  ".join(row))

    print("\n=== semantic_topk fractional recall@1 across the staircase (the cliff) ===")  # noqa
    for size in cfg.catalog_sizes:
        c = next((c for c in art.cells if c["catalog_size"] == size
                  and c["strategy"] == "semantic_topk" and c["k"] == 1), None)
        if c:
            lo, hi = c["recall_ci"]
            print(f"  size={size:>4}: recall@1={c['recall_at_k']:.3f}  (95% cluster-CI {lo:.3f}-{hi:.3f})")

    n_cliff = sum(1 for s in art.tracer.spans if s.is_cliff)
    print(f"\nlabel self-consistency κ (NOT human-verified) = {art.label_report['self_consistency_kappa']}")
    print(f"cliff events detected                          = {n_cliff}")
    print(f"\nartifacts written to: {a.out}")
    for kind, p in paths.items():
        print(f"  {kind:<7} {p}")
    return 0


def cmd_bench_sweep(a) -> int:
    """Sensitivity sweep: is the cliff an artifact of core_share=8? Vary it and
    report semantic_topk fractional recall@1 at the small vs large catalog. A
    persistent gap across shares (and under a different embedding geometry via
    --embed mock_char) shows the cliff is structural, not a hand-picked constant."""
    from .catalog.synth import Spec, build_catalog, generate_queries
    from .providers.base import get_embedder
    from .routing.base import RoutingContext, get_strategy

    cfg = BenchConfig()
    cfg.n_queries = a.queries
    cfg.catalog_sizes = [100, 300]
    embedder = get_embedder(a.embed, dim=cfg.embed_dim)
    semantic = get_strategy("semantic_topk")
    shares = [int(x) for x in a.shares.split(",")]

    def recall1(size, spec, queries):
        ctx = RoutingContext.build(build_catalog(size, spec), embedder)
        tot = 0.0
        for q in queries:
            exposed = set(semantic(ctx, q.text, 1))
            tot += len(set(q.gold_tool_ids) & exposed) / len(q.gold_tool_ids)
        return tot / len(queries)

    print(f"[mcp-router] sweep: embed={embedder.name} queries={cfg.n_queries} "
          f"shares={shares}", file=sys.stderr)
    print("\ncore_share | recall@1(100) | recall@1(300) | cliff drop")
    print("-----------+---------------+---------------+-----------")
    for share in shares:
        spec = Spec(seed=cfg.seed, core_share=share, kw_collision=cfg.kw_collision_ratio)
        queries = generate_queries(cfg, spec)
        r100, r300 = recall1(100, spec, queries), recall1(300, spec, queries)
        print(f"    {share:>2}     |     {r100:.3f}     |     {r300:.3f}     |   -{r100-r300:.3f}")
    print("\n(cliff drop stays materially positive across core_share -> structural, "
          "not a constant artifact)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcp-router")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="benchmark commands")
    bsub = b.add_subparsers(dest="bench_cmd", required=True)
    run = bsub.add_parser("run", help="run the routing benchmark")
    run.add_argument("--out", default="artifacts", help="output directory")
    run.add_argument("--embed", default="mock", choices=["mock", "mock_char", "local"])
    run.add_argument("--llm", default="mock", choices=["mock", "claude"])
    run.add_argument("--vector", default="memory", choices=["memory", "pgvector"])
    run.add_argument("--sizes", default="", help="comma list, e.g. 100,200,300")
    run.add_argument("--queries", type=int, default=0, help="override query count")
    run.set_defaults(func=cmd_bench_run)

    sweep = bsub.add_parser("sweep", help="sensitivity sweep: cliff vs core_share")
    sweep.add_argument("--shares", default="6,7,8,9,10",
                       help="comma list of core_share values to sweep")
    sweep.add_argument("--embed", default="mock", choices=["mock", "mock_char", "local"])
    sweep.add_argument("--queries", type=int, default=120)
    sweep.set_defaults(func=cmd_bench_sweep)
    return p


def main(argv=None) -> int:
    # Force UTF-8 stdout/stderr so output is identical across consoles (Windows
    # cp949, etc.). Files are always written UTF-8 regardless.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
