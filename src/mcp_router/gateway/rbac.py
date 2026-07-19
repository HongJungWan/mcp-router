"""Per-tenant RBAC over the federated catalog.

A tenant policy is allow/deny glob patterns matched against a tool's
namespaced_name (e.g. "github.create_issue") and its server ("github").
Rules: deny always wins; an empty allow-list means "allow everything not denied".
Unknown tenants fall back to a configurable default (default: allow all).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase   # case-sensitive => same decision on Windows and POSIX
from typing import Dict, List


@dataclass
class TenantPolicy:
    name: str
    allow: List[str] = field(default_factory=list)   # empty => allow all (minus deny)
    deny: List[str] = field(default_factory=list)

    def permits(self, tool) -> bool:
        targets = (tool.namespaced_name, tool.group)
        if any(fnmatchcase(t, p) for p in self.deny for t in targets):
            return False
        if not self.allow:
            return True
        return any(fnmatchcase(t, p) for p in self.allow for t in targets)


@dataclass
class Rbac:
    policies: Dict[str, TenantPolicy] = field(default_factory=dict)
    default: TenantPolicy = field(default_factory=lambda: TenantPolicy("default"))

    def policy(self, tenant: str) -> TenantPolicy:
        return self.policies.get(tenant, self.default)

    def permits(self, tenant: str, tool) -> bool:
        return self.policy(tenant).permits(tool)

    def filter(self, tenant: str, tools):
        p = self.policy(tenant)
        return [t for t in tools if p.permits(t)]

    @classmethod
    def from_config(cls, cfg: dict) -> "Rbac":
        """cfg = {"tenants": {name: {"allow": [...], "deny": [...]}}, "default": {...}}"""
        pols = {n: TenantPolicy(n, v.get("allow", []), v.get("deny", []))
                for n, v in (cfg.get("tenants") or {}).items()}
        dflt_cfg = cfg.get("default") or {}
        dflt = TenantPolicy("default", dflt_cfg.get("allow", []), dflt_cfg.get("deny", []))
        return cls(policies=pols, default=dflt)
