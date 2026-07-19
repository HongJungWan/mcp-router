"""Single-shot tool-selection agent (NOT a ReAct loop).

The agent receives ONLY the tools the gateway exposed and must pick the right
one(s) in a single step — there is no reason/act/observe iteration. If routing
dropped the gold tool, the agent literally cannot select it; that is how a recall
miss becomes a task failure.

It is NOT told how many gold tools exist (no cardinality leak): it selects up to
`budget` tools and self-limits below that via the llm's confidence gap. The same
agent wraps either the offline mock (Jaccard) llm or the real Claude tool-use llm
(both expose `choose_tools`) — so no separate agent framework is needed.
"""
from __future__ import annotations

from typing import List

from ..models import Tool

_BUDGET = 3


class SelectionAgent:
    def __init__(self, llm, budget: int = _BUDGET):
        self.llm = llm
        self.budget = budget
        self.model_id = getattr(llm, "model_id", "selector")

    def run(self, query: str, exposed_tools: List[Tool]) -> List[int]:
        return self.llm.choose_tools(query, exposed_tools, self.budget)


def get_agent(llm, budget: int = _BUDGET) -> SelectionAgent:
    return SelectionAgent(llm, budget)
