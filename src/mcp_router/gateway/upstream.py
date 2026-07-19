"""Upstream MCP server abstraction.

`MockUpstream` is the offline default: in-process, backed by the harvested real
tool definitions (data/real_mcp_tools.json), with a switchable `fail` flag so the
circuit breaker can be exercised deterministically. Real stdio/HTTP MCP upstreams
implement the same tiny surface and are the opt-in path (see roadmap).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Protocol, runtime_checkable


class UpstreamError(RuntimeError):
    """An upstream tool call failed (network/protocol/tool error)."""


@runtime_checkable
class Upstream(Protocol):
    name: str
    def list_tools(self) -> List[Dict]: ...            # [{"name","description"}, ...]
    def call_tool(self, name: str, arguments: dict) -> dict: ...


class MockUpstream:
    def __init__(self, name: str, tools: List[Dict], fail: bool = False):
        self.name = name
        self._tools = tools
        self.fail = fail          # flip to True to simulate an outage (breaker demo)
        self.calls = 0

    def list_tools(self) -> List[Dict]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls += 1
        if self.fail:
            raise UpstreamError(f"upstream '{self.name}' is down")
        return {"ok": True, "server": self.name, "tool": name, "echo": arguments}


def upstreams_from_dataset(path: str | None = None) -> List[MockUpstream]:
    """Build one MockUpstream per server in the harvested dataset."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "data", "real_mcp_tools.json")
    data = json.load(open(path, encoding="utf-8"))["tools"]
    by_server: Dict[str, List[Dict]] = {}
    for t in data:
        by_server.setdefault(t["server"], []).append(
            {"name": t["name"], "description": t["description"]})
    return [MockUpstream(s, tools) for s, tools in sorted(by_server.items())]
