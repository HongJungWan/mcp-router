"""Deterministic, offline mock providers.

Two embedding geometries are provided so the recall cliff can be shown to be a
property of *near-duplicate crowding*, not an artifact of one embedding:

* MockEmbedder      hashed bag-of-words (token-additive; cosine ~ shared tokens)
* MockCharEmbedder  hashed char-trigrams (subword-smoothed; still overlap-correlated)

Independence (the anti-"closed-loop" fix): the three actors that used to share
one `cos + 2·lex` formula are now deliberately DIFFERENT functions —
  - router (hybrid)  : embedding cosine + exact rare-keyword lexical
  - agent (MockLLM)  : full-description token Jaccard + deterministic noise
                       (NO keyword super-weight)  -> selection competence
  - labeler          : semantic + lexical with its OWN threshold cardinality
So `task_success` is no longer a re-labeling of recall: even when routing exposes
the gold tool, the agent scores it by a different signal and can still be fooled
by an exposed near-duplicate. A recall *miss* still forces a failure (the agent
never sees the tool), but a recall *hit* does not trivially imply success.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import EMBED_DIM
from ..determinism import jaccard, stable_hash, tokenize
from ..vectorstore.base import l2_normalize


class MockEmbedder:
    name = "mock-bow-hash-v1"

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim or EMBED_DIM
        self._cache: dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        vec = [0.0] * self.dim
        for t in set(tokenize(text)):
            vec[stable_hash("tok:" + t) % self.dim] += 1.0
        vec = l2_normalize(vec)
        self._cache[text] = vec
        return vec


class MockCharEmbedder:
    """Hashed character-trigram embedding: a subword-smoothed variant whose cosine
    still correlates strongly with token overlap (empirically close to BoW). It is
    a cheap robustness check that the cliff is not an artifact of the *exact* BoW
    tokenizer — it is NOT proof of independence from lexical overlap and NOT a
    semantic embedding. Real dense-embedding validation (bge-small) is still owed."""
    name = "mock-chartrigram-v1"

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim or EMBED_DIM
        self._cache: dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        s = " " + text.lower() + " "
        vec = [0.0] * self.dim
        for g in {s[i:i + 3] for i in range(len(s) - 2)}:
            vec[stable_hash("tri:" + g) % self.dim] += 1.0
        vec = l2_normalize(vec)
        self._cache[text] = vec
        return vec


class MockLLM:
    """Deterministic stand-in for a tool-use model — the AGENT's selection brain.

    Intentionally decoupled from the router: scores candidates by Jaccard overlap
    of the full tokenized descriptions (plus tiny deterministic noise), with NO
    rare-keyword super-weight. This is a different competence model than routing,
    so task-success is an independent signal layered on top of recall.
    """
    model_id = "mock-jaccard-agent-v1"
    # A candidate is selected if within this fraction of the top score. The agent
    # thus decides ITS OWN count (up to the budget n) — it is never told |gold|.
    _GAP = 0.10

    def __init__(self, embedder=None):
        self.embedder = embedder  # kept for API symmetry; unused for selection

    def choose_tools(self, query: str, candidates: list, n: int) -> List[int]:
        if not candidates:
            return []
        qset = set(tokenize(query))
        scored = []
        for tool in candidates:
            base = jaccard(qset, set(tokenize(tool.description)))
            noise = (stable_hash(f"sel|{query}|{tool.id}") % 1000) / 1e6
            scored.append((tool.id, base + noise))
        scored.sort(key=lambda t: (-t[1], t[0]))
        top = scored[0][1]
        thresh = top - self._GAP * (abs(top) + 1e-9)
        picked = [tid for tid, s in scored[:n] if s >= thresh]
        return picked or [scored[0][0]]
