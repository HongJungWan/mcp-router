"""Lightweight span recorder.

Records, per (query, strategy, k, catalog_size), the candidate/exposed sets and
the gold tool's rank in the full semantic ranking — so a reader can see *the
exact query where top-k dropped the gold tool to rank k+1*. That is the "recall
cliff trace" the project is built to expose. Pure-stdlib; writes
newline-delimited JSON. (An OpenTelemetry emitter was intentionally cut: a
single-process offline batch bench has no collector and no distributed serving,
so JSONL is sufficient — see README "over-engineering" note.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from .determinism import stable_hash


def make_trace_id(*parts: object) -> str:
    return f"{stable_hash('|'.join(str(p) for p in parts)):016x}"


@dataclass
class Span:
    trace_id: str
    query_id: int
    strategy: str
    k: int
    catalog_size: int
    difficulty: str
    gold_tool_ids: List[int]
    gold_ranks: List[int]          # rank of each gold in full semantic ranking (1-based)
    exposed_count: int
    candidate_count: int
    recall_hit: bool
    task_success: bool
    exposed_token_cost: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @property
    def is_cliff(self) -> bool:
        """A cliff event: gold exists but was ranked just past the cutoff k."""
        return (not self.recall_hit) and any(r > self.k for r in self.gold_ranks)


@dataclass
class Tracer:
    spans: List[Span] = field(default_factory=list)

    def record(self, span: Span) -> None:
        self.spans.append(span)

    def flush(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in self.spans:
                f.write(s.to_json() + "\n")

    def _semantic_cliffs(self) -> List[Span]:
        # gold_ranks are the SEMANTIC full-ranking positions, so a cliff event is
        # only meaningful for semantic_topk spans. Counting other strategies'
        # spans here would misattribute semantic ranks and ~2x inflate the count.
        return [s for s in self.spans if s.strategy == "semantic_topk" and s.is_cliff]

    def n_cliff(self) -> int:
        return len(self._semantic_cliffs())

    def cliff_events(self, limit: int = 20) -> List[Span]:
        return self._semantic_cliffs()[:limit]
