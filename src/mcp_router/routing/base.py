"""Routing context (prebuilt indexes) + strategy registry.

A strategy maps (query, k) -> the ordered list of tool ids the gateway would
expose to the model. recall@k and token cost are computed from that exposed set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from ..determinism import tokenize
from ..models import Catalog
from ..vectorstore.base import VectorIndex, get_index


@dataclass
class RoutingContext:
    catalog: Catalog
    embedder: object                       # EmbeddingProvider
    tool_index: VectorIndex
    group_vectors: Dict[str, List[float]]  # group id -> embedding
    _by_id: dict = field(default_factory=dict)

    @classmethod
    def build(cls, catalog: Catalog, embedder, backend: str = "memory") -> "RoutingContext":
        idx = get_index(backend, dim=getattr(embedder, "dim", 512))
        for t in catalog.tools:
            idx.add(t.id, embedder.embed(t.embed_text))
        gvecs = {g.name: embedder.embed(g.description) for g in catalog.groups}
        return cls(catalog=catalog, embedder=embedder, tool_index=idx,
                   group_vectors=gvecs, _by_id=catalog.by_id())


# registry --------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[RoutingContext, str, int], List[int]]] = {}


def strategy(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_strategy(name: str) -> Callable[[RoutingContext, str, int], List[int]]:
    if name not in _REGISTRY:
        raise ValueError(f"unknown strategy '{name}'. known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


STRATEGIES = ["passthrough", "semantic_topk", "hierarchical", "hybrid"]

# import concrete strategies to populate the registry (side-effect import)
from . import strategies as _strategies  # noqa: E402,F401
