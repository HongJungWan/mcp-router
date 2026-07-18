"""Pure-Python in-memory vector index. Deterministic exact cosine search.

This is the offline default. Ties are broken by item_id so ordering is stable
across runs/machines (critical for reproducible recall@k)."""
from __future__ import annotations

from typing import List, Tuple

from .base import VectorIndex, cosine


class MemoryIndex(VectorIndex):
    def __init__(self, dim: int = 512):
        self.dim = dim
        self._vecs: dict[int, List[float]] = {}

    def add(self, item_id: int, vector: List[float]) -> None:
        self._vecs[item_id] = vector

    def search(self, vector: List[float], k: int,
               allowed: set[int] | None = None) -> List[Tuple[int, float]]:
        ids = self._vecs.keys() if allowed is None else (i for i in self._vecs if i in allowed)
        scored = [(i, cosine(vector, self._vecs[i])) for i in ids]
        # sort by score desc, then id asc for deterministic tie-breaking
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[: k if k is not None else len(scored)]
