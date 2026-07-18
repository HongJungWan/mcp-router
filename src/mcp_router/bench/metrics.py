"""Metrics: recall@k, task success, token/latency stats, bootstrap CIs, McNemar.

Pure stdlib. Bootstrap uses a seeded RNG so CIs are reproducible.
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
    """Percentile bootstrap CI for the mean of a 0/1 (or real) sample."""
    if not values:
        return (0.0, 0.0)
    r = rng(seed_key, "bootstrap", len(values), n_resamples)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        resample = [values[r.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = percentile(means, 100 * (alpha / 2))
    hi = percentile(means, 100 * (1 - alpha / 2))
    return (round(lo, 4), round(hi, 4))


def mcnemar(success_a: Sequence[int], success_b: Sequence[int]) -> Dict[str, float]:
    """Paired comparison of two strategies' per-query task success (0/1).
    Returns discordant counts, the (continuity-corrected) chi-square statistic,
    and its p-value under 1 dof (P = erfc(sqrt(x/2)))."""
    b = sum(1 for x, y in zip(success_a, success_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(success_a, success_b) if x == 0 and y == 1)
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": round(chi2, 4), "p_value": round(p, 5)}


def recall_at_k(recall_hits: Sequence[bool]) -> float:
    return mean([1.0 if h else 0.0 for h in recall_hits])


def success_rate(successes: Sequence[bool]) -> float:
    return mean([1.0 if s else 0.0 for s in successes])
