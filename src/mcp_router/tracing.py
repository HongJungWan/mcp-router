"""Lightweight span recorder.

Records, per (query, strategy, k, catalog_size), the candidate/exposed sets and
the gold tool's rank in the full semantic ranking — so a reader can see *the
exact query where top-k dropped the gold tool to rank k+1*. That is the "recall
cliff trace" the project is built to expose.

If `opentelemetry` is installed and MCPR_OTEL=1, spans are also emitted via
OTLP; otherwise this stays pure-stdlib and writes newline-delimited JSON.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

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
    _otel: Optional[object] = None

    def __post_init__(self):
        if os.environ.get("MCPR_OTEL") == "1":
            try:  # pragma: no cover - optional
                from opentelemetry import trace
                self._otel = trace.get_tracer("mcp_router")
            except Exception:
                self._otel = None

    def record(self, span: Span) -> None:
        self.spans.append(span)
        if self._otel is not None:  # pragma: no cover - optional
            with self._otel.start_as_current_span("route") as s:
                for kk, vv in span.__dict__.items():
                    s.set_attribute(f"mcp_router.{kk}", vv if isinstance(vv, (int, float, bool, str)) else str(vv))

    def flush(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in self.spans:
                f.write(s.to_json() + "\n")

    def cliff_events(self, limit: int = 20) -> List[Span]:
        return [s for s in self.spans if s.is_cliff][:limit]
