"""Vector index behind a stable interface.

`memory` (pure-Python exact cosine, offline, deterministic). A pgvector/HNSW
backend was intentionally NOT built: for a few-hundred-tool catalog, exact
brute-force cosine is both faster and exact, so ANN would be scale cosplay.
"""
from .base import VectorIndex, cosine, dot, l2_normalize, get_index

__all__ = ["VectorIndex", "cosine", "dot", "l2_normalize", "get_index"]
