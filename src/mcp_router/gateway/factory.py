"""Build a Gateway from a JSON config (stdlib — no YAML dependency).

config = {
  "strategy": "hybrid",
  "breaker": {"failure_threshold": 3, "reset_timeout": 30.0},
  "rbac": {...},
  "upstreams": [                       # optional; default = in-process mocks from the dataset
    {"type": "stdio", "name": "everything", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-everything"]},
    {"type": "http", "name": "remote", "url": "http://localhost:9000/mcp"}
  ]
}
Real stdio/http upstreams use the official MCP SDK (pip install .[mcp]).
"""
from __future__ import annotations

import json
from typing import List, Optional

from .federation import Federation
from .rbac import Rbac
from .server import Gateway
from .upstream import Upstream, upstreams_from_dataset


def _require(spec: dict, key: str):
    if key not in spec:
        raise ValueError(f"upstream {spec.get('name', '?')!r}: missing required field {key!r}")
    return spec[key]


def _build_upstreams(cfg: dict) -> List[Upstream]:
    specs = cfg.get("upstreams")
    if not specs:                                   # default: offline mocks from the dataset
        return upstreams_from_dataset(cfg.get("dataset"))
    out: List[Upstream] = []
    try:
        for s in specs:
            kind = s.get("type", "stdio")
            if kind == "stdio":
                from .mcp_upstream import stdio_upstream    # lazy: needs .[mcp]
                out.append(stdio_upstream(_require(s, "name"), _require(s, "command"),
                                          s.get("args"), s.get("env"), s.get("cwd")))
            elif kind == "http":
                from .mcp_upstream import http_upstream
                out.append(http_upstream(_require(s, "name"), _require(s, "url"), s.get("headers")))
            else:
                raise ValueError(f"unknown upstream type: {kind}")
    except Exception:
        for u in out:                               # don't orphan already-started upstreams
            try:
                u.close()
            except Exception:
                pass
        raise
    return out


def build_gateway(config_path: Optional[str] = None) -> Gateway:
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    fed = Federation(_build_upstreams(cfg))
    rbac = Rbac.from_config(cfg.get("rbac", {}))
    return Gateway(fed, rbac, strategy=cfg.get("strategy", "hybrid"),
                   breaker_kwargs=cfg.get("breaker"))
