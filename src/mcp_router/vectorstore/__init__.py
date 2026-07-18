"""Vector index behind a stable interface.

Default = `memory` (pure-Python cosine, offline, deterministic).
Opt-in `pgvector` backend (HNSW in Postgres) via pip install .[pg].
"""
from .base import VectorIndex, cosine, dot, l2_normalize, get_index

__all__ = ["VectorIndex", "cosine", "dot", "l2_normalize", "get_index"]
