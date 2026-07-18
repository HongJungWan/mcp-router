"""The load-bearing test: the recall cliff must be real and reproducible, and
hybrid must recover what semantic-topk loses. Pure stdlib (unittest)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_router.config import BenchConfig
from mcp_router.catalog import build_catalog, generate_queries
from mcp_router.providers.base import get_embedder, get_llm
from mcp_router.routing.base import RoutingContext, get_strategy


def _recall_at_k(strategy_name, size, k, queries, ctx):
    strat = get_strategy(strategy_name)
    hits = 0
    for q in queries:
        exposed = set(strat(ctx, q.text, k))
        if set(q.gold_tool_ids).issubset(exposed):
            hits += 1
    return hits / len(queries)


class CliffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = BenchConfig()
        cls.cfg.n_queries = 120
        cls.emb = get_embedder("mock", dim=cls.cfg.embed_dim)
        cls.queries = generate_queries(cls.cfg)
        cls.ctx = {s: RoutingContext.build(build_catalog(s), cls.emb)
                   for s in (100, 300)}

    def test_catalog_is_nested_staircase(self):
        small = build_catalog(100).tools
        big = build_catalog(300).tools
        self.assertEqual([t.id for t in small], [t.id for t in big[:100]])

    def test_recall_cliff_semantic_topk(self):
        r100 = _recall_at_k("semantic_topk", 100, 1, self.queries, self.ctx[100])
        r300 = _recall_at_k("semantic_topk", 300, 1, self.queries, self.ctx[300])
        # recall@1 must degrade materially as the distractor pool grows.
        self.assertGreater(r100, r300 + 0.15, f"no cliff: {r100:.2f} -> {r300:.2f}")

    def test_hybrid_recovers_recall(self):
        s = _recall_at_k("semantic_topk", 300, 3, self.queries, self.ctx[300])
        h = _recall_at_k("hybrid", 300, 3, self.queries, self.ctx[300])
        self.assertGreater(h, s + 0.1, f"hybrid did not recover: {s:.2f} vs {h:.2f}")

    def test_passthrough_perfect_recall(self):
        r = _recall_at_k("passthrough", 300, 1, self.queries, self.ctx[300])
        self.assertEqual(r, 1.0)

    def test_recall_monotonic_in_k(self):
        prev = -1.0
        for k in (1, 3, 5, 10):
            r = _recall_at_k("semantic_topk", 300, k, self.queries, self.ctx[300])
            self.assertGreaterEqual(r + 1e-9, prev)
            prev = r


if __name__ == "__main__":
    unittest.main(verbosity=2)
