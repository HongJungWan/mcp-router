"""Cohen's kappa for two binary labelers (stdlib only)."""
from __future__ import annotations

from typing import List, Tuple


def cohen_kappa(a: List[int], b: List[int]) -> float:
    """kappa = (po - pe) / (1 - pe) for two 0/1 label sequences of equal length."""
    if len(a) != len(b):
        raise ValueError("label sequences must be equal length")
    n = len(a)
    if n == 0:
        return 1.0
    # confusion counts
    n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    n00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    po = (n11 + n00) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement(a: List[int], b: List[int]) -> float:
    if not a:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)
