"""Build a Gateway from a JSON config (stdlib — no YAML dependency).

config = {
  "strategy": "hybrid",
  "breaker": {"failure_threshold": 3, "reset_timeout": 30.0},
  "rbac": {"tenants": {"ci": {"allow": ["github.*", "filesystem.*"]}},
           "default": {"deny": ["*.delete_*", "mongodb.drop-*"]}}
}
Upstreams default to in-process mocks from the harvested dataset.
"""
from __future__ import annotations

import json
from typing import Optional

from .federation import Federation
from .rbac import Rbac
from .server import Gateway
from .upstream import upstreams_from_dataset


def build_gateway(config_path: Optional[str] = None) -> Gateway:
    cfg = json.load(open(config_path, encoding="utf-8")) if config_path else {}
    fed = Federation(upstreams_from_dataset(cfg.get("dataset")))
    rbac = Rbac.from_config(cfg.get("rbac", {}))
    return Gateway(fed, rbac, strategy=cfg.get("strategy", "hybrid"),
                   breaker_kwargs=cfg.get("breaker"))
