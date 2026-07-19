"""Core domain models. Plain dataclasses (stdlib) — no ORM required."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Tool:
    id: int
    namespaced_name: str          # e.g. "email.send_smtp"
    group: str                    # topic/group id, e.g. "email"
    description: str
    keywords: List[str]           # discriminating tokens (for lexical match)
    is_distractor: bool = False   # near-duplicate crowd tool
    token_cost: int = 0           # estimated schema tokens exposed to the LLM
    input_schema: Optional[dict] = None   # JSON Schema from a real upstream (None for synthetic)

    @property
    def embed_text(self) -> str:
        return f"{self.namespaced_name} {self.description}"


@dataclass(frozen=True)
class ToolGroup:
    name: str
    description: str
    tool_ids: List[int]


@dataclass(frozen=True)
class Query:
    id: int
    text: str
    gold_tool_ids: List[int]                 # ground truth (we synthesize it)
    group: str
    difficulty: str                          # single | multi | ambiguous
    distractor_pool_size: int                # catalog size this query is asked against


@dataclass
class Catalog:
    size: int
    tools: List[Tool]
    groups: List[ToolGroup]

    def by_id(self) -> dict[int, Tool]:
        return {t.id: t for t in self.tools}


@dataclass
class RouteOutcome:
    """Result of routing a single query under one strategy at one k."""
    query_id: int
    strategy: str
    k: int
    exposed_tool_ids: List[int]
    exposed_token_cost: int
    recall_hit: bool                         # ALL gold tools present (set-cover / hit-rate)
    recall_fraction: float                   # |gold ∩ exposed| / |gold|  (fractional recall@k)
    difficulty: str                          # single | multi | ambiguous (for stratification)
    cluster: int                             # gold-tool id used as bootstrap cluster key
    # task-execution (single-shot selection agent) outcome:
    selected_tool_ids: List[int] = field(default_factory=list)
    task_success: bool = False
    trace_id: str = ""
