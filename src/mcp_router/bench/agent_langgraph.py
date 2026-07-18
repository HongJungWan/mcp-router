"""LangGraph ReAct agent — production counterpart to MockReActAgent.

MockReActAgent (bench/agent.py) is the offline-deterministic default used by the
pure-stdlib path. This module is the real, opt-in implementation: it drives an
actual LangGraph ReAct loop over Claude tool-use, exposing each routed Tool as a
callable the model may invoke. Which tools the model calls *is* its selection.

Heavy, optional dependencies (``langgraph``, ``langchain_anthropic``) are
imported INSIDE ``__init__`` so that merely importing this module never breaks
the stdlib default path. Enable via ``pip install .[agent]`` and an
``ANTHROPIC_API_KEY`` in the environment.
"""
from __future__ import annotations

import re
from typing import List

from ..models import Tool


class LangGraphReActAgent:
    """Real ReAct agent over Claude tool-use, mirroring MockReActAgent's API.

    ``run(query, exposed_tools, n_gold) -> list[int]`` returns up to ``n_gold``
    ids of the tools the model actually invoked, in first-invocation order.
    """

    def __init__(self, llm=None):
        # Heavy deps imported here (not at module top) so importing this module
        # never breaks the offline stdlib default path.
        from langchain_anthropic import ChatAnthropic  # noqa: F401
        from langgraph.prebuilt import create_react_agent  # noqa: F401

        self.llm = llm
        # Use the injected LLM's model id when provided so the agent and the
        # ClaudeLLM labeler/router share one model config (and one VCR cache),
        # instead of silently hardcoding a second model.
        self.model_id = getattr(llm, "model_id", None) or "claude-sonnet-5"
        self._ChatAnthropic = ChatAnthropic
        self._create_react_agent = create_react_agent

    @staticmethod
    def _tool_fn_name(tool: Tool) -> str:
        """A safe function name the LLM tool-use API accepts (^[a-zA-Z0-9_-]+$)."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool.namespaced_name)
        return f"{name}__{tool.id}"

    def run(self, query: str, exposed_tools: List[Tool], n_gold: int) -> List[int]:
        if not exposed_tools or n_gold <= 0:
            return []

        from langchain_core.tools import StructuredTool

        selected: List[int] = []

        def _make_tool(t: Tool):
            fn_name = self._tool_fn_name(t)

            def _record(**_kwargs) -> str:
                if t.id not in selected:
                    selected.append(t.id)
                return f"invoked {t.namespaced_name}"

            return StructuredTool.from_function(
                func=_record,
                name=fn_name,
                description=t.description or t.namespaced_name,
            )

        tools = [_make_tool(t) for t in exposed_tools]

        model = self._ChatAnthropic(model=self.model_id, temperature=0)
        agent = self._create_react_agent(model, tools)

        system = (
            "You are a tool-routing agent. From the available tools, invoke the "
            f"{n_gold} tool(s) that best satisfy the user's request. Call each "
            "chosen tool exactly once. Do not invoke tools that are irrelevant."
        )
        messages = [("system", system), ("user", query)]

        try:
            agent.invoke({"messages": messages})
        except Exception:
            # Robustness: a failed run yields whatever was recorded before the
            # error (possibly nothing) rather than crashing the bench.
            pass

        # Clip to n_gold: the model may under- or over-select.
        return selected[:n_gold]
