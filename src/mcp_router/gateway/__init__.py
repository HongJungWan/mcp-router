"""M1 — the serving gateway that M3 evaluated offline.

Federates upstream MCP servers into one namespaced catalog, applies per-tenant
RBAC, routes/​budgets the exposed tool set (reusing the M3 router), and guards
every upstream call with a circuit breaker.

Pure-stdlib and deterministic: the default upstreams are in-process mocks backed
by the harvested real tool definitions (data/real_mcp_tools.json), and the
breaker takes an injected clock, so the whole gateway runs and tests offline with
no network. Real stdio/HTTP MCP upstreams are the opt-in path.
"""
from .breaker import CircuitBreaker, CircuitOpenError, BreakerState
from .rbac import Rbac, TenantPolicy
from .federation import Federation
from .upstream import Upstream, MockUpstream
from .server import Gateway

__all__ = [
    "CircuitBreaker", "CircuitOpenError", "BreakerState",
    "Rbac", "TenantPolicy", "Federation", "Upstream", "MockUpstream", "Gateway",
]
