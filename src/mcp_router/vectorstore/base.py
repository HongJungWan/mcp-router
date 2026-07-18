"""Vector math (pure Python) + VectorIndex protocol + factory."""
from __future__ import annotations

import math
from typing import List, Protocol, Tuple, runtime_checkable


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return list(v)
    return [x / n for x in v]


def cosine(a: List[float], b: List[float]) -> float:
    # vectors are stored normalized, but normalize defensively.
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


@runtime_checkable
class VectorIndex(Protocol):
    def add(self, item_id: int, vector: List[float]) -> None: ...
    def search(self, vector: List[float], k: int,
               allowed: set[int] | None = None) -> List[Tuple[int, float]]:
        """Return up to k (item_id, score) sorted by score desc. If `allowed`
        is given, restrict the candidate set to those ids (group routing)."""
        ...


def get_index(backend: str = "memory", dim: int = 512) -> "VectorIndex":
    if backend == "memory":
        from .memory import MemoryIndex
        return MemoryIndex(dim=dim)
    # A production pgvector/HNSW backend is a roadmap item. For catalogs of a few
    # hundred tools, exact pure-Python cosine is faster and exact than ANN, so it
    # was intentionally not built (see README "over-engineering" note).
    raise ValueError(f"unknown/unsupported vector backend: {backend} (only 'memory')")
