"""Tests for the interview-hardening fixes:
  * agent selection is DECOUPLED from the router (no keyword super-weight)
  * distractors collide on the gold keyword (hybrid can lose -> non-degenerate)
  * the cliff survives a different embedding geometry (char-trigram)
  * the labeler reports self-consistency, not 'human-verified', and leaks no n
  * BH correction + cluster bootstrap behave
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_router.config import BenchConfig
from mcp_router.catalog import build_catalog, generate_queries
from mcp_router.labeling import label_quality_report
from mcp_router.models import Tool
from mcp_router.providers.base import get_embedder, get_llm
from mcp_router.providers.mock import MockLLM
from mcp_router.routing.base import RoutingContext, get_strategy
from mcp_router.bench.metrics import benjamini_hochberg, cluster_bootstrap_ci


def _recall1(embed, size, spec, queries):
    ctx = RoutingContext.build(build_catalog(size, spec), embed)
    sem = get_strategy("semantic_topk")
    tot = 0.0
    for q in queries:
        ex = set(sem(ctx, q.text, 1))
        tot += len(set(q.gold_tool_ids) & ex) / len(q.gold_tool_ids)
    return tot / len(queries)


class AgentDecouplingTest(unittest.TestCase):
    def test_agent_ignores_keyword_uses_description_overlap(self):
        # A: no query keyword, but high description overlap. B: has the query
        # keyword, but unrelated description. The DECOUPLED agent must pick A —
        # proving it does not use the router's rare-keyword super-weight.
        q = "please send a email message with alpha beta gamma using smtp7"
        a = Tool(0, "x.a", "email", "send a email message with alpha beta gamma delta",
                 ["zzz"], False, 50)
        b = Tool(1, "x.b", "email", "totally unrelated omega sigma tau lambda", ["smtp7"], False, 50)
        self.assertEqual(MockLLM().choose_tools(q, [a, b], 1), [0])


class KeywordCollisionTest(unittest.TestCase):
    def test_some_distractors_collide_on_gold_keyword(self):
        cat = build_catalog(300)
        base_kw = {t.keywords[0] for t in cat.tools if not t.is_distractor}
        collisions = [t for t in cat.tools if t.is_distractor and t.keywords[0] in base_kw]
        self.assertGreater(len(collisions), 0, "no keyword-collision distractors -> McNemar stays degenerate")


class CliffGeometryTest(unittest.TestCase):
    def test_cliff_survives_char_trigram_embedding(self):
        cfg = BenchConfig(); cfg.n_queries = 120
        from mcp_router.catalog.synth import DEFAULT_SPEC
        emb = get_embedder("mock_char", dim=cfg.embed_dim)
        qs = generate_queries(cfg, DEFAULT_SPEC)
        r100 = _recall1(emb, 100, DEFAULT_SPEC, qs)
        r300 = _recall1(emb, 300, DEFAULT_SPEC, qs)
        self.assertGreater(r100, r300 + 0.1,
                           f"cliff vanished under char geometry: {r100:.2f} -> {r300:.2f}")


class LabelerHonestyTest(unittest.TestCase):
    def test_report_is_self_consistency_not_human_verified(self):
        cfg = BenchConfig(); cfg.n_queries = 120
        emb = get_embedder("mock", dim=cfg.embed_dim)
        llm = get_llm("mock", embedder=emb)
        rep = label_quality_report(build_catalog(300), generate_queries(cfg), llm, cfg)
        self.assertNotIn("human_verified_n", rep)
        self.assertIn("NOT human", rep["note"])
        self.assertIn("self_consistency_kappa", rep)
        self.assertTrue(0.0 <= rep["self_consistency_kappa"] <= 1.0)


class StatsTest(unittest.TestCase):
    def test_bh_monotone_and_bounded(self):
        q = benjamini_hochberg([0.001, 0.02, 0.04, 0.5])
        self.assertTrue(all(0.0 <= x <= 1.0 for x in q))
        self.assertLessEqual(q[0], q[3])

    def test_cluster_bootstrap_deterministic(self):
        vals = [1.0, 0.0, 1.0, 1.0, 0.0] * 6
        clusters = [i % 5 for i in range(30)]
        a = cluster_bootstrap_ci(vals, clusters, 300, "seedkey")
        b = cluster_bootstrap_ci(vals, clusters, 300, "seedkey")
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
