"""Unit tests for metrics, kappa, embedder determinism, and routing invariants."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_router.labeling.kappa import cohen_kappa
from mcp_router.bench.metrics import mcnemar, percentile, bootstrap_ci
from mcp_router.providers.mock import MockEmbedder
from mcp_router.vectorstore.base import cosine
from mcp_router.vectorstore.memory import MemoryIndex


class KappaTest(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertAlmostEqual(cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_chance_agreement_is_zero_ish(self):
        a = [1, 0] * 10
        b = [1, 1, 0, 0] * 5
        self.assertLess(abs(cohen_kappa(a, b)), 0.35)


class MetricsTest(unittest.TestCase):
    def test_percentile(self):
        xs = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(xs, 50), 3)
        self.assertEqual(percentile(xs, 0), 1)
        self.assertEqual(percentile(xs, 100), 5)

    def test_mcnemar_symmetry(self):
        a = [1, 1, 0, 0, 1]
        b = [0, 1, 1, 0, 1]
        r = mcnemar(a, b)
        self.assertIn("p_value", r)
        self.assertTrue(0.0 <= r["p_value"] <= 1.0)

    def test_bootstrap_ci_deterministic(self):
        vals = [1.0, 0.0, 1.0, 1.0, 0.0] * 10
        a = bootstrap_ci(vals, 500, "k")
        b = bootstrap_ci(vals, 500, "k")
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])


class EmbedderTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        e = MockEmbedder(dim=256)
        v1 = e.embed("send an email using smtp")
        v2 = e.embed("send an email using smtp")
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(sum(x * x for x in v1) ** 0.5, 1.0, places=6)

    def test_more_shared_tokens_higher_cosine(self):
        e = MockEmbedder(dim=512)
        q = e.embed("alpha beta gamma delta epsilon")
        near = e.embed("alpha beta gamma delta zeta")     # 4 shared
        far = e.embed("alpha beta omega sigma tau")        # 2 shared
        self.assertGreater(cosine(q, near), cosine(q, far))


class IndexTest(unittest.TestCase):
    def test_tie_break_by_id(self):
        idx = MemoryIndex(dim=4)
        idx.add(5, [1.0, 0, 0, 0])
        idx.add(2, [1.0, 0, 0, 0])
        res = idx.search([1.0, 0, 0, 0], k=2)
        self.assertEqual([r[0] for r in res], [2, 5])  # equal score -> lower id first

    def test_allowed_filter(self):
        idx = MemoryIndex(dim=4)
        for i in range(5):
            idx.add(i, [1.0, 0, 0, 0])
        res = idx.search([1.0, 0, 0, 0], k=10, allowed={1, 3})
        self.assertEqual(sorted(r[0] for r in res), [1, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
