"""ReAct-style tool-using agent.

The agent receives ONLY the tools the gateway exposed and must pick the right
one(s). If routing dropped the gold tool, the agent literally cannot select it —
that is how a recall miss becomes a task failure.

Default = MockReActAgent (deterministic, offline). Opt-in LangGraphReActAgent
(pip install .[agent], ANTHROPIC_API_KEY) implements the same interface with a
real LangGraph ReAct loop over Claude tool-use.
"""
from __future__ import annotations

from typing import List

from ..models import Tool


class MockReActAgent:
    model_id = "mock-react-v1"

    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str, exposed_tools: List[Tool], n_gold: int) -> List[int]:
        # A single deterministic reason+act step: score exposed tools, select n.
        return self.llm.choose_tools(query, exposed_tools, n_gold)


def get_agent(kind: str, llm):
    if kind == "mock":
        return MockReActAgent(llm)
    if kind == "langgraph":  # pragma: no cover - optional heavy dep
        from .agent_langgraph import LangGraphReActAgent
        return LangGraphReActAgent(llm)
    raise ValueError(f"unknown agent kind: {kind}")
