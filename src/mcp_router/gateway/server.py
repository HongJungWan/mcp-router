"""The gateway: federation + RBAC + routing + circuit breaker in one object.

- tools/list(tenant, query): RBAC-filter the federated catalog, then (if a query
  hint is given) route it through an M3 strategy so only a budgeted top-k is
  exposed; without a query, return the whole allowed set.
- tools/call(tenant, name, args): RBAC-check, resolve to the upstream, and make
  the call through that upstream's circuit breaker.

Deterministic and offline by default (mock upstreams + mock embedder + injected
clock). The transport (HTTP JSON-RPC) is a thin shell over this object.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

from ..models import Catalog, ToolGroup
from ..providers.base import get_embedder
from ..routing.base import RoutingContext, get_strategy
from .breaker import CircuitBreaker
from .federation import Federation
from .rbac import Rbac

_CTX_CACHE_MAX = 64          # bounded: keyed by (policy identity, config_hash)


class Gateway:
    def __init__(self, federation: Federation, rbac: Optional[Rbac] = None,
                 embedder=None, strategy: str = "hybrid",
                 breaker_kwargs: Optional[dict] = None,
                 now: Callable[[], float] = time.monotonic):
        self.fed = federation
        self.rbac = rbac or Rbac()
        self.embedder = embedder or get_embedder("mock")
        self.strategy_name = strategy
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._bkw = breaker_kwargs or {}
        self._now = now
        self._ctx_cache: Dict[tuple, RoutingContext] = {}
        self._lock = threading.Lock()   # guards _breakers / _ctx_cache (ThreadingHTTPServer)

    # ---- internals -----------------------------------------------------------
    def _breaker(self, upstream_name: str) -> CircuitBreaker:
        with self._lock:               # atomic get-or-create (no duplicate breakers)
            return self._breakers.setdefault(
                upstream_name, CircuitBreaker(upstream_name, now=self._now, **self._bkw))

    def _ctx(self, tenant: str) -> RoutingContext:
        # Key by the tenant's *effective policy identity*, not the raw (attacker-
        # controlled, unauthenticated) X-Tenant string — unknown tenants collapse
        # to the default policy, so the cache is bounded by (#policies + config).
        key = (self.rbac.policy(tenant).name, self.fed.config_hash)
        with self._lock:
            ctx = self._ctx_cache.get(key)
        if ctx is not None:
            return ctx
        cat = self.fed.catalog()
        allowed = self.rbac.filter(tenant, cat.tools)
        gmap: Dict[str, List[int]] = {}
        for t in allowed:
            gmap.setdefault(t.group, []).append(t.id)
        gdesc = {g.name: g.description for g in cat.groups}
        groups = [ToolGroup(name=g, description=gdesc.get(g, g), tool_ids=ids)
                  for g, ids in gmap.items()]
        sub = Catalog(size=len(allowed), tools=allowed, groups=groups)
        ctx = RoutingContext.build(sub, self.embedder)
        with self._lock:
            if len(self._ctx_cache) >= _CTX_CACHE_MAX:   # simple bound; drop stale hashes
                self._ctx_cache.clear()
            self._ctx_cache[key] = ctx
        return ctx

    # ---- MCP-ish surface -----------------------------------------------------
    def list_tools(self, tenant: str = "default", query: str = "",
                   k: int = 10, strategy: Optional[str] = None) -> List[dict]:
        ctx = self._ctx(tenant)
        by_id = ctx.catalog.by_id()
        if not query:
            ids = [t.id for t in ctx.catalog.tools]           # no hint -> full allowed set
        else:
            ids = get_strategy(strategy or self.strategy_name)(ctx, query, k)
        return [{"name": by_id[i].namespaced_name, "description": by_id[i].description,
                 "inputSchema": by_id[i].input_schema or {"type": "object"}} for i in ids]

    def call_tool(self, tenant: str, name: str, arguments: Optional[dict] = None) -> dict:
        cat = self.fed.catalog()
        tool = next((t for t in cat.tools if t.namespaced_name == name), None)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        if not self.rbac.permits(tenant, tool):
            raise PermissionError(f"tenant '{tenant}' is not permitted to call '{name}'")
        upstream, local = self.fed.resolve(name)
        breaker = self._breaker(upstream.name)
        return breaker.call(lambda: upstream.call_tool(local, arguments or {}))

    def stats(self) -> dict:
        cat = self.fed.catalog()
        return {
            "config_hash": self.fed.config_hash,
            "upstreams": sorted(self.fed.upstreams),
            "n_tools": len(cat.tools),
            "breakers": [b.snapshot() for b in self._breakers.values()],
        }
