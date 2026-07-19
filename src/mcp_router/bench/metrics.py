"""Statistics: percentile, iid + cluster bootstrap CIs, McNemar (with p
formatting), Benjamini-Hochberg correction, and phi correlation. Pure stdlib;
bootstraps use a seeded RNG so CIs are reproducible. (recall@k and token
aggregation live in runner.py, not here.)
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ..determinism import rng


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = p / 100.0 * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def bootstrap_ci(values: Sequence[float], n_resamples: int, seed_key: str,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 (or real) sample (iid)."""
    if not values:
        return (0.0, 0.0)
    r = rng(seed_key, "bootstrap", len(values), n_resamples)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[r.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (round(percentile(means, 100 * alpha / 2), 4),
            round(percentile(means, 100 * (1 - alpha / 2)), 4))


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[int],
                         n_resamples: int, seed_key: str, alpha: float = 0.05) -> Tuple[float, float]:
    """Cluster (block) bootstrap: resample whole gold-tool clusters, not single
    queries. The queries cluster over the gold tools, so an iid bootstrap
    understates the CI; resampling clusters respects that dependence. Note: a
    query is keyed to a single gold tool (its lowest gold id), so multi-gold
    queries' cross-cluster dependence is only partly modelled — the multi
    stratum CI can be slightly anti-conservative."""
    if not values:
        return (0.0, 0.0)
    groups: Dict[int, List[float]] = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    keys = list(groups)
    r = rng(seed_key, "clusterboot", len(values), n_resamples)
    means = []
    for _ in range(n_resamples):
        pool: List[float] = []
        for _ in range(len(keys)):
            pool.extend(groups[keys[r.randrange(len(keys))]])
        means.append(sum(pool) / len(pool) if pool else 0.0)
    means.sort()
    return (round(percentile(means, 100 * alpha / 2), 4),
            round(percentile(means, 100 * (1 - alpha / 2)), 4))


def format_p(p: float) -> str:
    if p <= 0:
        return "<1e-300"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """BH-adjusted p-values (q-values) for multiple-comparison control."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * m / (rank + 1))
        q[i] = min(prev, 1.0)
    return q


def mcnemar(success_a: Sequence[int], success_b: Sequence[int]) -> Dict[str, float]:
    """Paired comparison of two strategies on a binary per-query outcome (0/1).
    The benchmark applies this to recall_hit (see runner._mcnemar_table), not to
    task_success. b = (a=1,b=0), c = (a=0,b=1). Continuity-corrected chi-square,
    1 dof; p kept as a raw float (not rounded to 0) plus a formatted string."""
    b = sum(1 for x, y in zip(success_a, success_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(success_a, success_b) if x == 0 and y == 1)
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "p_str": "1.0000"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": round(chi2, 4), "p_value": p, "p_str": format_p(p)}


def phi_correlation(x: Sequence[int], y: Sequence[int]) -> float:
    """Phi (Matthews) correlation between two binary sequences. Used to report
    whether task_success is genuinely independent of recall_hit (low |phi|) or
    just a re-labeling of it (high |phi|)."""
    n = len(x)
    if n == 0:
        return 0.0
    n11 = sum(1 for a, b in zip(x, y) if a and b)
    n10 = sum(1 for a, b in zip(x, y) if a and not b)
    n01 = sum(1 for a, b in zip(x, y) if not a and b)
    n00 = sum(1 for a, b in zip(x, y) if not a and not b)
    num = n11 * n00 - n10 * n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return round(num / den, 4) if den else 0.0
