"""Real end-to-end MCP integration over the official SDK (stdio subprocesses,
no network). Skipped automatically if `mcp` isn't installed.

- client bridge: our McpUpstream connects to a real FastMCP server subprocess.
- server transport: the SDK client (our McpUpstream) connects to OUR gateway
  served over MCP stdio — proving both directions of the SDK integration.
"""
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HAS_MCP = importlib.util.find_spec("mcp") is not None
FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@unittest.skipUnless(HAS_MCP, "official MCP SDK not installed (pip install .[mcp])")
class McpClientBridgeTest(unittest.TestCase):
    def test_stdio_upstream_roundtrip(self):
        from mcp_router.gateway.mcp_upstream import stdio_upstream
        up = stdio_upstream("mini", sys.executable, [os.path.join(FIX, "mini_mcp_server.py")])
        try:
            self.assertEqual({t["name"] for t in up.list_tools()}, {"echo", "add"})
            r = up.call_tool("echo", {"text": "hello-mcp"})
            self.assertTrue(r["ok"])
            self.assertIn("hello-mcp", r["content"])
            self.assertTrue(up.call_tool("add", {"a": 2, "b": 3})["ok"])
        finally:
            up.close()

    def test_gateway_federates_real_upstream(self):
        from mcp_router.gateway import Federation, Gateway, Rbac
        from mcp_router.gateway.mcp_upstream import stdio_upstream
        up = stdio_upstream("mini", sys.executable, [os.path.join(FIX, "mini_mcp_server.py")])
        try:
            gw = Gateway(Federation([up]), Rbac())
            self.assertEqual({t["name"] for t in gw.list_tools()}, {"mini.echo", "mini.add"})
            self.assertTrue(gw.call_tool("default", "mini.echo", {"text": "x"})["ok"])
        finally:
            up.close()


@unittest.skipUnless(HAS_MCP, "official MCP SDK not installed")
class McpServerTransportTest(unittest.TestCase):
    def test_gateway_served_over_mcp_stdio(self):
        # Our gateway, served over the official MCP stdio transport, hit by the SDK client.
        from mcp_router.gateway.mcp_upstream import stdio_upstream
        client = stdio_upstream("gw", sys.executable, [os.path.join(FIX, "gateway_stdio_server.py")])
        try:
            self.assertEqual({t["name"] for t in client.list_tools()}, {"demo.ping", "demo.status"})
            r = client.call_tool("demo.ping", {})
            self.assertTrue(r["content"])          # gateway returns json-encoded upstream result
        finally:
            client.close()


@unittest.skipUnless(HAS_MCP, "official MCP SDK not installed")
class McpLifecycleTest(unittest.TestCase):
    def test_init_failure_reports_real_cause(self):
        from mcp_router.gateway.mcp_upstream import stdio_upstream
        from mcp_router.gateway.upstream import UpstreamError
        with self.assertRaises(UpstreamError) as cm:
            stdio_upstream("bad", "this_command_does_not_exist_xyz123", [])
        self.assertNotIn("initialize: None", str(cm.exception))  # captured the true cause

    def test_close_is_idempotent(self):
        from mcp_router.gateway.mcp_upstream import stdio_upstream
        up = stdio_upstream("mini", sys.executable, [os.path.join(FIX, "mini_mcp_server.py")])
        up.close()
        up.close()   # second close must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
