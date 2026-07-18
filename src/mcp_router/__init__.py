"""mcp_router — MCP Gateway tool-routing benchmark harness (M3).

The default execution path is pure-stdlib and deterministic so `make bench`
reproduces byte-identical results offline, with no API keys or network.
Production adapters (Claude, LangGraph, pgvector) live behind the same
interfaces and are opt-in via extras + environment variables.
"""

__version__ = "0.1.0"
