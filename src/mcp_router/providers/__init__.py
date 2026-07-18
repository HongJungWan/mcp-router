"""Embedding + LLM providers behind stable interfaces.

Default = deterministic `mock` (offline, no deps). Opt-in real adapters:
  - providers.claude.ClaudeLLM         (pip install .[claude], ANTHROPIC_API_KEY)
  - providers.local_embed.LocalEmbed   (pip install .[local])
"""
from .base import EmbeddingProvider, LLMProvider, get_embedder, get_llm

__all__ = ["EmbeddingProvider", "LLMProvider", "get_embedder", "get_llm"]
