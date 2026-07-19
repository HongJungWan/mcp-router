"""A thin stdlib HTTP JSON-RPC 2.0 transport over the Gateway.

MCP speaks JSON-RPC; this exposes `tools/list`, `tools/call`, and a
`gateway/stats` extension so the gateway is an actually-runnable server with no
third-party deps. Tenant is taken from the `X-Tenant` header (default: "default").
The official MCP SDK (stdio/streamable-HTTP) is the production transport — this
is the offline, dependency-free equivalent for demos and tests.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .breaker import CircuitOpenError
from .upstream import UpstreamError


class _Handler(BaseHTTPRequestHandler):
    gateway = None  # injected per-server subclass

    def log_message(self, *_):        # keep the server quiet
        pass

    def _tenant(self) -> str:
        return self.headers.get("X-Tenant", "default")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send({"jsonrpc": "2.0", "id": None,
                               "error": {"code": -32700, "message": "parse error"}})
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "result": self._dispatch(method, params)})
        except PermissionError as e:            # RBAC denial
            self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": str(e)}})
        except CircuitOpenError as e:           # upstream tripped
            self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32002, "message": str(e)}})
        except UpstreamError as e:              # server-side: upstream call failed
            self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}})
        except KeyError as e:                   # client-side: unknown tool/method
            self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": str(e)}})
        except Exception as e:                  # pragma: no cover
            self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}})

    def _dispatch(self, method, params):
        gw = self.gateway
        if method == "tools/list":
            return {"tools": gw.list_tools(self._tenant(), params.get("query", ""),
                                           params.get("k", 10), params.get("strategy"))}
        if method == "tools/call":
            return gw.call_tool(self._tenant(), params["name"], params.get("arguments") or {})
        if method == "gateway/stats":
            return gw.stats()
        raise KeyError(f"unknown method: {method}")

    def _send(self, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(gateway, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (_Handler,), {"gateway": gateway})
    return ThreadingHTTPServer((host, port), handler)
