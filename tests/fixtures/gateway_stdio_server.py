"""Serve a small mock-backed Gateway as a real MCP server over stdio (official
SDK). The integration test connects to this via the SDK client to prove the
server transport works end to end. Run: python gateway_stdio_server.py"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_router.gateway import Federation, Gateway, Rbac
from mcp_router.gateway.upstream import MockUpstream
from mcp_router.gateway.mcp_server import run_stdio

_gw = Gateway(Federation([MockUpstream("demo", [
    {"name": "ping", "description": "ping the demo server"},
    {"name": "status", "description": "get demo server status"},
])]), Rbac())

if __name__ == "__main__":
    asyncio.run(run_stdio(_gw, tenant="default"))
