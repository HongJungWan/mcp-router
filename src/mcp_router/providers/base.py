"""Provider protocols + a factory that selects mock (default) or real adapters."""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int
    def embed(self, text: str) -> List[float]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal 'select tools for this query' surface used by the selection agent and
    the labeler. Real adapters implement the same method via tool-use / function
    calling; the mock implements it deterministically."""
    model_id: str
    def choose_tools(self, query: str, candidates: list, n: int) -> List[int]:
        """Return up to `n` tool ids chosen from `candidates` (list[Tool])."""
        ...


def get_embedder(kind: str = "mock", dim: int | None = None) -> EmbeddingProvider:
    if kind == "mock":
        from .mock import MockEmbedder
        return MockEmbedder(dim=dim)
    if kind == "mock_char":                       # different geometry, robustness check
        from .mock import MockCharEmbedder
        return MockCharEmbedder(dim=dim)
    if kind == "local":
        from .local_embed import LocalEmbed
        return LocalEmbed()
    raise ValueError(f"unknown embed provider: {kind}")


def get_llm(kind: str = "mock", embedder: EmbeddingProvider | None = None) -> LLMProvider:
    if kind == "mock":
        from .mock import MockLLM
        return MockLLM(embedder=embedder)
    if kind == "claude":
        from .claude import ClaudeLLM
        return ClaudeLLM()
    raise ValueError(f"unknown llm provider: {kind}")
