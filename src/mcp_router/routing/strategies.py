"""The four routing strategies.

passthrough     expose everything (perfect recall, unaffordable tokens) — baseline
semantic_topk   global vector top-k (loses recall as distractors crowd)
hierarchical    route to top-2 groups, then vector top-k within them
hybrid          vector + lexical keyword match (recovers recall the keyword pins)
"""
from __future__ import annotations

from typing import List

from ..determinism import tokenize
from ..vectorstore.base import cosine
from .base import RoutingContext, strategy

HIER_GROUPS = 2          # groups kept by hierarchical routing
LEX_WEIGHT = 2.0         # lexical dominates ties (a matched rare kw beats any cosine)


@strategy("passthrough")
def passthrough(ctx: RoutingContext, query: str, k: int) -> List[int]:
    # k is ignored: everything is exposed. This is the recall ceiling / token floor.
    return [t.id for t in ctx.catalog.tools]


@strategy("semantic_topk")
def semantic_topk(ctx: RoutingContext, query: str, k: int) -> List[int]:
    q = ctx.embedder.embed(query)
    return [tid for tid, _ in ctx.tool_index.search(q, k)]


@strategy("hierarchical")
def hierarchical(ctx: RoutingContext, query: str, k: int) -> List[int]:
    q = ctx.embedder.embed(query)
    group_scores = sorted(
        ((cosine(q, v), g) for g, v in ctx.group_vectors.items()),
        key=lambda t: (-t[0], t[1]),
    )
    keep = {g for _, g in group_scores[:HIER_GROUPS]}
    allowed = {t.id for t in ctx.catalog.tools if t.group in keep}
    return [tid for tid, _ in ctx.tool_index.search(q, k, allowed=allowed)]


@strategy("hybrid")
def hybrid(ctx: RoutingContext, query: str, k: int) -> List[int]:
    q = ctx.embedder.embed(query)
    qtokens = set(tokenize(query))
    # full semantic ranking, then fold in exact lexical keyword match
    sem = ctx.tool_index.search(q, k=None)
    scored = []
    for tid, cos in sem:
        tool = ctx._by_id[tid]
        kws = tool.keywords or []
        lex = sum(1 for w in kws if w in qtokens) / len(kws) if kws else 0.0
        scored.append((tid, cos + LEX_WEIGHT * lex))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [tid for tid, _ in scored[:k]]
