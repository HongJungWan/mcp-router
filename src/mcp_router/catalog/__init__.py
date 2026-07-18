"""Synthetic tool-catalog builder with a staircase distractor pool.

The offline default is fully synthetic + deterministic. The production path
(harvesting real MCP servers: filesystem/github/slack/postgres/brave-search) is
a documented roadmap item; the harness interface is identical either way.
"""
from .synth import (
    build_catalog, query_tokens, generate_queries, HOT_TARGETS, N_BASE,
)

__all__ = [
    "build_catalog", "query_tokens", "generate_queries", "HOT_TARGETS", "N_BASE",
]
