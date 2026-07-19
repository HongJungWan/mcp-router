"""Federate N upstream MCP servers into one namespaced catalog.

Each upstream tool becomes `server.tool`; that prefix makes names unique across
servers (a guard still catches an intra-server duplicate). Produces a `Catalog`
(same type M3 routes over) so the gateway can reuse the routing strategies, and
a resolver mapping a namespaced name back to (upstream, local tool name).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..catalog.synth import estimate_tokens
from ..determinism import stable_hash
from ..models import Catalog, Tool, ToolGroup
from .upstream import Upstream


class Federation:
    def __init__(self, upstreams: List[Upstream]):
        self.upstreams: Dict[str, Upstream] = self._index(upstreams)
        self._build()

    @staticmethod
    def _index(upstreams: List[Upstream]) -> Dict[str, Upstream]:
        names = [u.name for u in upstreams]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:                          # silent overwrite would drop a server's tools
            raise ValueError(f"duplicate upstream names: {dups}")
        return {u.name: u for u in upstreams}

    def _build(self) -> None:
        tools: List[Tool] = []
        resolve: Dict[str, Tuple[str, str]] = {}
        groups: Dict[str, List[int]] = {}
        seen = set()
        tid = 0
        for uname, u in self.upstreams.items():
            for t in u.list_tools():
                local = t["name"]
                ns = f"{uname}.{local}"
                if ns in seen:                      # intra-server duplicate guard
                    ns = f"{ns}__{tid}"
                seen.add(ns)
                desc = t.get("description", "")
                tools.append(Tool(id=tid, namespaced_name=ns, group=uname,
                                   description=desc, keywords=[], is_distractor=False,
                                   token_cost=estimate_tokens(ns, desc)))
                resolve[ns] = (uname, local)
                groups.setdefault(uname, []).append(tid)
                tid += 1
        self._tools = tools
        self._resolve = resolve
        self._groups = [
            ToolGroup(name=g, description=f"{g} server tools: "
                      + " ".join(self._tools[i].description for i in ids)[:400],
                      tool_ids=ids)
            for g, ids in groups.items()
        ]
        # digest of tool CONTENT (not just server names) so a reload that changes
        # tools — same server names — produces a new hash and invalidates caches.
        digest = "|".join(f"{t.namespaced_name}:{t.description}:{t.token_cost}" for t in tools)
        self.config_hash = f"{stable_hash(digest):x}"

    def catalog(self) -> Catalog:
        return Catalog(size=len(self._tools), tools=list(self._tools), groups=list(self._groups))

    def resolve(self, namespaced_name: str) -> Tuple[Upstream, str]:
        if namespaced_name not in self._resolve:
            raise KeyError(f"unknown tool: {namespaced_name}")
        uname, local = self._resolve[namespaced_name]
        return self.upstreams[uname], local

    def reload(self, upstreams: List[Upstream]) -> None:
        self.upstreams = self._index(upstreams)
        self._build()
