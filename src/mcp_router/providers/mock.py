"""Deterministic, offline mock providers.

Design intent — this is what makes the benchmark *demonstrate* a real recall
cliff without any network or model:

* MockEmbedder is a hashed bag-of-words embedding. Near-duplicate distractor
  tools share almost every token with the true tool and the query, so their
  cosine is nearly identical. As the catalog (distractor pool) grows, distractors
  crowd the true tool out of a small top-k => recall@k drops. That drop is a
  genuine property of the geometry, not a scripted number.

* MockLLM.choose_tools scores the *exposed* candidates by embedding cosine only
  (no lexical signal). So even when routing exposes the gold tool, an embedding-
  close distractor can still be picked => task can fail. This reproduces the
  causal chain: routing that drops recall -> the agent literally cannot pick the
  right tool -> task failure.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import EMBED_DIM
from ..determinism import stable_hash, tokenize
from ..vectorstore.base import cosine, l2_normalize


class MockEmbedder:
    name = "mock-bow-hash-v1"

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim or EMBED_DIM
        self._cache: dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        vec = [0.0] * self.dim
        toks = tokenize(text)
        # Additive bag-of-words: cosine is (near-)monotonic in shared-token count,
        # which is what makes recall@k a clean, explainable function of how many
        # near-duplicate distractors out-share the true tool with the query.
        for t in set(toks):  # set() -> presence, not frequency; stabilizes ranking
            idx = stable_hash("tok:" + t) % self.dim
            vec[idx] += 1.0
        vec = l2_normalize(vec)
        self._cache[text] = vec
        return vec


class MockLLM:
    """Deterministic stand-in for a tool-use / function-calling model."""
    model_id = "mock-react-v1"

    def __init__(self, embedder: Optional[MockEmbedder] = None):
        self.embedder = embedder or MockEmbedder()

    def choose_tools(self, query: str, candidates: list, n: int) -> List[int]:
        # A competent tool-user reads each exposed tool's schema, so selection is
        # semantic + exact-keyword (same signal a real Claude tool-use agent has).
        # Consequence: if routing EXPOSED the gold tool, the agent picks it for
        # keyword-bearing queries; if routing dropped it, the agent literally
        # cannot -> a recall miss becomes a task failure. Ambiguous queries (no
        # keyword) still trip selection even when the tool is exposed.
        if not candidates:
            return []
        q = self.embedder.embed(query)
        qtokens = set(tokenize(query))
        scored = []
        for tool in candidates:
            lex = sum(1 for w in (tool.keywords or []) if w in qtokens)
            s = cosine(q, self.embedder.embed(tool.embed_text)) + 2.0 * lex
            scored.append((tool.id, s))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [tid for tid, _ in scored[:n]]
