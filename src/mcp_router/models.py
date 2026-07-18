"""Core domain models. Plain dataclasses (stdlib) — no ORM required for the
offline bench; the pgvector backend maps these to tables 1:1."""
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

    def group_of(self, tool_id: int) -> str:
        return self.by_id()[tool_id].group


@dataclass
class RouteOutcome:
    """Result of routing a single query under one strategy at one k."""
    query_id: int
    strategy: str
    k: int
    exposed_tool_ids: List[int]
    exposed_token_cost: int
    recall_hit: bool                         # all gold tools present in exposed set
    latency_ms: float
    # task-execution (ReAct agent) outcome:
    selected_tool_ids: List[int] = field(default_factory=list)
    task_success: bool = False
    trace_id: str = ""


@dataclass
class BenchRun:
    """Reproducibility envelope: every knob that shaped the numbers."""
    git_sha: str
    seed: int
    embed_model: str
    llm_model_id: str
    vector_backend: str
    config: dict
    strategy: str
    k: int
    catalog_size: int
