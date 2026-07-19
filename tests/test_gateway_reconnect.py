"""Reconnect / health-check state machine, tested WITHOUT the MCP SDK via an
injected fake session + fake transport. Deterministic (tiny intervals)."""
import contextlib
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_router.gateway.mcp_upstream import McpUpstream
from mcp_router.gateway.upstream import UpstreamError


class _Tool:
    def __init__(self, name, desc):
        self.name, self.description, self.inputSchema = name, desc, {"type": "object"}


class _ListResult:
    def __init__(self, tools): self.tools = tools


class _CallResult:
    content, structuredContent, isError = [], None, False


class FakeSession:
    """die_after_pings=0 => never dies; N => the (N+1)-th ping raises (session dead)."""
    def __init__(self, die_after_pings=0):
        self._die, self._pings = die_after_pings, 0

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def initialize(self): pass
    async def list_tools(self): return _ListResult([_Tool("t1", "one")])
    async def call_tool(self, name, args): return _CallResult()

    async def send_ping(self):
        self._pings += 1
        if self._die and self._pings > self._die:
            raise RuntimeError("session dead")


@contextlib.asynccontextmanager
async def fake_connect():
    yield (object(), object())


def _wait(pred, secs=3.0):
    end = time.time() + secs
    while time.time() < end and not pred():
        time.sleep(0.02)


class ReconnectTest(unittest.TestCase):
    def test_reconnects_after_session_death_and_recovers(self):
        connects = []

        def sf(streams):
            n = len(connects)
            connects.append(1)
            return FakeSession(die_after_pings=1 if n == 0 else 0)   # first dies, then healthy

        up = McpUpstream("fake", fake_connect, session_factory=sf,
                         health_interval=0.02, backoff_base=0.01, backoff_cap=0.05)
        try:
            self.assertTrue(up.health()["healthy"])
            _wait(lambda: up.health()["reconnects"] >= 1)
            self.assertGreaterEqual(up.health()["reconnects"], 1)   # detected death, reconnected
            _wait(lambda: up.health()["healthy"])
            self.assertTrue(up.health()["healthy"])                 # recovered on the new session
            self.assertGreaterEqual(len(connects), 2)
            self.assertEqual([t["name"] for t in up.list_tools()], ["t1"])  # usable again
        finally:
            up.close()

    def test_close_stops_the_loop(self):
        up = McpUpstream("fake", fake_connect, session_factory=lambda s: FakeSession(0),
                         health_interval=0.02)
        up.close()
        with self.assertRaises(UpstreamError):
            up.list_tools()               # session gone after close

    def test_init_failure_surfaces_cause(self):
        @contextlib.asynccontextmanager
        async def bad_connect():
            raise RuntimeError("cannot connect")
            yield  # unreachable; makes this a valid async generator

        with self.assertRaises(UpstreamError) as cm:
            McpUpstream("bad", bad_connect, session_factory=lambda s: FakeSession(0),
                        init_timeout=1.5, max_retries=1, backoff_base=0.01)
        self.assertIn("cannot connect", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
