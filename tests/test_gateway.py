"""Gateway (M1) tests: circuit breaker, RBAC, federation, end-to-end, transport.
Pure stdlib; the breaker uses an injected clock so cooldown is deterministic."""
import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_router.models import Tool
from mcp_router.gateway.breaker import CircuitBreaker, BreakerState, CircuitOpenError
from mcp_router.gateway.rbac import Rbac, TenantPolicy
from mcp_router.gateway.federation import Federation
from mcp_router.gateway.upstream import MockUpstream, UpstreamError
from mcp_router.gateway.server import Gateway
from mcp_router.gateway.transport import make_server


def _tool(ns, group, desc="do a thing with data and options"):
    return Tool(id=0, namespaced_name=ns, group=group, description=desc,
                keywords=[], is_distractor=False, token_cost=10)


def _fed():
    a = MockUpstream("alpha", [{"name": "read", "description": "read a record from alpha store"},
                               {"name": "write", "description": "write a record to alpha store"}])
    b = MockUpstream("beta", [{"name": "read", "description": "read a document from beta service"},
                              {"name": "search", "description": "search documents in beta service"}])
    return Federation([a, b]), a, b


class BreakerTest(unittest.TestCase):
    def test_trip_open_halfopen_close(self):
        clk = [0.0]
        br = CircuitBreaker("x", failure_threshold=2, reset_timeout=10.0, now=lambda: clk[0])
        boom = lambda: (_ for _ in ()).throw(ValueError("boom"))
        self.assertIs(br.state, BreakerState.CLOSED)
        for _ in range(2):
            with self.assertRaises(ValueError):
                br.call(boom)
        self.assertIs(br.state, BreakerState.OPEN)
        with self.assertRaises(CircuitOpenError):
            br.call(lambda: 1)
        clk[0] = 11.0                                   # cooldown elapsed
        self.assertIs(br.state, BreakerState.HALF_OPEN)
        self.assertEqual(br.call(lambda: 42), 42)       # probe succeeds -> close
        self.assertIs(br.state, BreakerState.CLOSED)
        self.assertEqual(br.trips, 1)

    def test_halfopen_probe_failure_reopens(self):
        clk = [0.0]
        br = CircuitBreaker("x", failure_threshold=1, reset_timeout=5.0, now=lambda: clk[0])
        with self.assertRaises(ValueError):
            br.call(lambda: (_ for _ in ()).throw(ValueError()))
        self.assertIs(br.state, BreakerState.OPEN)
        clk[0] = 6.0
        self.assertIs(br.state, BreakerState.HALF_OPEN)
        with self.assertRaises(ValueError):             # probe fails
            br.call(lambda: (_ for _ in ()).throw(ValueError()))
        clk[0] = 6.5                                    # still within new cooldown
        self.assertIs(br.state, BreakerState.OPEN)


class RbacTest(unittest.TestCase):
    def test_allow_deny_precedence(self):
        p = TenantPolicy("t", allow=["github.*"], deny=["github.delete_*"])
        self.assertTrue(p.permits(_tool("github.create_issue", "github")))
        self.assertFalse(p.permits(_tool("github.delete_repo", "github")))  # deny wins
        self.assertFalse(p.permits(_tool("slack.post", "slack")))           # not allowed
        empty = TenantPolicy("t2", deny=["*.drop-*"])
        self.assertTrue(empty.permits(_tool("mongodb.find", "mongodb")))    # empty allow => all
        self.assertFalse(empty.permits(_tool("mongodb.drop-database", "mongodb")))

    def test_filter_and_default(self):
        rbac = Rbac(policies={"ci": TenantPolicy("ci", allow=["alpha.*"])},
                    default=TenantPolicy("default"))
        tools = [_tool("alpha.read", "alpha"), _tool("beta.read", "beta")]
        self.assertEqual([t.namespaced_name for t in rbac.filter("ci", tools)], ["alpha.read"])
        self.assertEqual(len(rbac.filter("unknown", tools)), 2)             # default allows all


class FederationTest(unittest.TestCase):
    def test_namespacing_and_resolve(self):
        fed, a, b = _fed()
        cat = fed.catalog()
        names = {t.namespaced_name for t in cat.tools}
        self.assertEqual(names, {"alpha.read", "alpha.write", "beta.read", "beta.search"})
        up, local = fed.resolve("beta.search")
        self.assertIs(up, b)
        self.assertEqual(local, "search")
        with self.assertRaises(KeyError):
            fed.resolve("nope.nope")


class GatewayE2ETest(unittest.TestCase):
    def test_list_call_rbac_breaker(self):
        fed, a, b = _fed()
        clk = [0.0]
        rbac = Rbac(policies={"ci": TenantPolicy("ci", allow=["alpha.*"])})
        gw = Gateway(fed, rbac, strategy="hybrid",
                     breaker_kwargs={"failure_threshold": 2, "reset_timeout": 10.0},
                     now=lambda: clk[0])
        listed = {t["name"] for t in gw.list_tools("ci")}
        self.assertEqual(listed, {"alpha.read", "alpha.write"})             # RBAC filtered
        self.assertTrue(gw.call_tool("ci", "alpha.read", {})["ok"])
        with self.assertRaises(PermissionError):
            gw.call_tool("ci", "beta.read", {})                            # denied
        a.fail = True
        for _ in range(2):
            with self.assertRaises(UpstreamError):
                gw.call_tool("ci", "alpha.read", {})
        with self.assertRaises(CircuitOpenError):
            gw.call_tool("ci", "alpha.read", {})                           # tripped


class TransportTest(unittest.TestCase):
    def _post(self, port, method, params, tenant="default"):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}",
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
            headers={"Content-Type": "application/json", "X-Tenant": tenant})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def test_http_jsonrpc(self):
        fed, a, b = _fed()
        rbac = Rbac(policies={"ci": TenantPolicy("ci", allow=["alpha.*"])})
        gw = Gateway(fed, rbac)
        httpd = make_server(gw, "127.0.0.1", 0)
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            r = self._post(port, "tools/list", {}, tenant="ci")
            self.assertEqual({t["name"] for t in r["result"]["tools"]}, {"alpha.read", "alpha.write"})
            r = self._post(port, "tools/call", {"name": "alpha.read", "arguments": {}}, tenant="ci")
            self.assertTrue(r["result"]["ok"])
            r = self._post(port, "tools/call", {"name": "beta.read"}, tenant="ci")
            self.assertEqual(r["error"]["code"], -32001)                    # RBAC denial
            r = self._post(port, "gateway/stats", {})
            self.assertIn("breakers", r["result"])
        finally:
            httpd.shutdown()


class HardeningTest(unittest.TestCase):
    def test_duplicate_upstream_name_raises(self):
        with self.assertRaises(ValueError):
            Federation([MockUpstream("dup", [{"name": "a", "description": "x"}]),
                        MockUpstream("dup", [{"name": "b", "description": "y"}])])

    def test_config_hash_tracks_tool_content(self):
        f1 = Federation([MockUpstream("s", [{"name": "a", "description": "one"}])])
        f2 = Federation([MockUpstream("s", [{"name": "a", "description": "TWO"}])])
        self.assertNotEqual(f1.config_hash, f2.config_hash)  # content change -> new hash

    def test_rbac_case_sensitive(self):
        # deny pattern with wrong case must NOT match (deterministic across OS)
        p = TenantPolicy("t", deny=["github.Delete_*"])
        self.assertTrue(p.permits(_tool("github.delete_repo", "github")))

    def test_readonly_config_is_actually_readonly(self):
        from mcp_router.gateway.factory import build_gateway
        cfg = os.path.join(os.path.dirname(__file__), "..", "deploy", "gateway.config.json")
        gw = build_gateway(cfg)
        # reads allowed
        self.assertTrue(gw.call_tool("readonly", "github.get_issue", {})["ok"])
        self.assertTrue(gw.call_tool("readonly", "sqlite.read_query", {})["ok"])
        # mutations denied — including the tricky ones the old denylist missed
        for mut in ("github.create_issue", "mongodb.delete-many", "sqlite.write_query",
                    "git.git_commit", "filesystem.write_file", "slack.slack_post_message"):
            with self.assertRaises(PermissionError, msg=mut):
                gw.call_tool("readonly", mut, {})


class TransportErrorTest(unittest.TestCase):
    def test_upstream_failure_is_server_error(self):
        a = MockUpstream("alpha", [{"name": "read", "description": "read a record"}], fail=True)
        gw = Gateway(Federation([a]), Rbac())
        httpd = make_server(gw, "127.0.0.1", 0)
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                 "params": {"name": "alpha.read"}}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                body = json.loads(r.read())
            self.assertEqual(body["error"]["code"], -32000)   # server-side, not -32602
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
