"""Real local embedding provider backed by sentence-transformers.

Uses BAAI/bge-small-en-v1.5 (384-dim). The heavy `sentence_transformers`
dependency is imported lazily inside __init__ so that importing this module
never breaks the pure-stdlib default path. The loaded model is cached on the
class so repeated instantiations reuse the same weights.
"""
from __future__ import annotations

from typing import ClassVar, List, Optional


class LocalEmbed:
    name: str = "bge-small-en-v1.5"
    dim: int = 384

    _MODEL_ID: ClassVar[str] = "BAAI/bge-small-en-v1.5"
    _model: ClassVar[Optional[object]] = None

    def __init__(self) -> None:
        # Lazy, in-method import: keeps the stdlib default path importable even
        # when sentence-transformers is not installed.
        if LocalEmbed._model is None:
            from sentence_transformers import SentenceTransformer

            LocalEmbed._model = SentenceTransformer(self._MODEL_ID)
        self._cache: dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        vec = LocalEmbed._model.encode(
            text,
            normalize_embeddings=True,
        )
        out = [float(x) for x in vec.tolist()]
        self._cache[text] = out
        return out
