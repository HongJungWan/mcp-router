"""Expose the Gateway as a real MCP server via the official SDK (stdio).

`tools/list` -> gateway.list_tools(tenant); `tools/call` -> gateway.call_tool(...).
RBAC denials / upstream failures raise, which the SDK returns to the client as an
isError tool result. This is the REAL (non-mock) SDK server transport, verified
by a round-trip test; it is not yet production-hardened (single fixed tenant, no
auth/graceful-shutdown). `mcp` is imported lazily (pip install .[mcp]).

The stdio server serves a single client, so the tenant is fixed per process
(from config/CLI). A multi-tenant streamable-HTTP server is a roadmap item.
"""
from __future__ import annotations

import json


def make_mcp_server(gateway, tenant: str = "default", name: str = "mcp-router"):
    from mcp.server.lowlevel import Server
    import mcp.types as types

    server = Server(name)

    @server.list_tools()
    async def _list_tools():
        return [types.Tool(name=t["name"], description=t["description"],
                           inputSchema=t.get("inputSchema") or {"type": "object"})
                for t in gateway.list_tools(tenant)]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None):
        # PermissionError (RBAC) / UpstreamError / CircuitOpenError propagate; the
        # SDK converts a raised handler into an isError CallToolResult.
        result = gateway.call_tool(tenant, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    return server


async def run_stdio(gateway, tenant: str = "default", name: str = "mcp-router") -> None:
    from mcp.server.stdio import stdio_server
    server = make_mcp_server(gateway, tenant, name)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
