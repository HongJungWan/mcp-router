"""Real MCP upstreams over the official SDK (stdio / streamable-HTTP).

The SDK is async and the gateway is sync, so each upstream owns a dedicated
asyncio loop in a background thread. `_serve()` enters the client transport +
ClientSession context managers, initializes, then parks on a stop event so the
session stays open; `list_tools`/`call_tool` submit coroutines onto that loop and
block for the result. Concurrent calls are safe: the single loop + the SDK's
request-id multiplexing serialize them (no extra lock needed).

Scope: this is the REAL (non-mock) SDK-backed upstream, verified by a stdio
round-trip in the tests. It is NOT production-hardened — there is no session
reconnect/health-check yet; a dead session surfaces as UpstreamError rather than
auto-recovering. `mcp` is imported lazily (pip install .[mcp]).
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, Dict, List, Optional

from .upstream import UpstreamError


class _LoopThread:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout)
        except FuturesTimeout:
            self._loop.call_soon_threadsafe(fut.cancel)   # don't leak a pending coroutine
            raise

    def spawn(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def call_soon(self, fn) -> None:
        self._loop.call_soon_threadsafe(fn)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():                   # deterministically free loop fds
            self._loop.close()


def _text(result) -> str:
    parts = [getattr(c, "text", "") for c in (result.content or []) if getattr(c, "type", "") == "text"]
    return "\n".join(p for p in parts if p)


class McpUpstream:
    """Wraps an open MCP ClientSession behind the sync Upstream interface."""

    def __init__(self, name: str, connect_factory: Callable, init_timeout: float = 30.0):
        self.name = name
        self._connect = connect_factory
        self._lt = _LoopThread()
        self._session = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._err: Optional[BaseException] = None
        self._closed = False
        self._fut = self._lt.spawn(self._serve())
        if not self._ready.wait(timeout=init_timeout) or self._session is None:
            if self._err is None:
                try:
                    self._err = self._fut.exception(timeout=5)   # blocks for the true cause
                except Exception as e:
                    self._err = e
            self.close()
            raise UpstreamError(f"mcp upstream '{name}' failed to initialize: {self._err}")

    async def _serve(self) -> None:
        from mcp import ClientSession
        self._stop = asyncio.Event()
        try:
            async with self._connect() as streams:
                read, write = streams[0], streams[1]   # stdio=(r,w); http=(r,w,get_session_id)
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as e:                          # capture BEFORE unblocking __init__
            self._err = e
            self._ready.set()
            raise
        finally:
            self._session = None                        # session is unusable once serve exits

    def _require(self):
        s = self._session
        if s is None:
            raise UpstreamError(f"upstream '{self.name}' session is not available")
        return s

    def list_tools(self) -> List[Dict]:
        s = self._require()
        try:
            res = self._lt.run(s.list_tools(), timeout=30)
        except FuturesTimeout:
            raise UpstreamError(f"upstream '{self.name}' list_tools timed out")
        return [{"name": t.name, "description": t.description or "",
                 "inputSchema": getattr(t, "inputSchema", None)} for t in res.tools]

    def call_tool(self, name: str, arguments: dict) -> dict:
        s = self._require()
        try:
            res = self._lt.run(s.call_tool(name, arguments or {}), timeout=60)
        except FuturesTimeout:
            raise UpstreamError(f"upstream '{self.name}' call '{name}' timed out")
        if getattr(res, "isError", False):
            raise UpstreamError(f"upstream '{self.name}' tool '{name}' errored: {_text(res)}")
        return {"ok": True, "server": self.name, "tool": name,
                "content": _text(res), "structured": res.structuredContent}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Let _serve resume and run its `async with` __aexit__ (terminate the stdio
        # child, close transport streams) BEFORE the loop is torn down.
        if self._stop is not None:
            self._lt.call_soon(self._stop.set)
        else:                                           # init hung before _stop existed
            self._lt.call_soon(self._fut.cancel)
        try:
            self._fut.result(timeout=10)
        except Exception:
            pass
        self._lt.stop()


def stdio_upstream(name: str, command: str, args: Optional[list] = None,
                   env: Optional[dict] = None, cwd: Optional[str] = None) -> McpUpstream:
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=command, args=args or [], env=env, cwd=cwd)
    return McpUpstream(name, lambda: stdio_client(params))


def http_upstream(name: str, url: str, headers: Optional[dict] = None) -> McpUpstream:
    from mcp.client.streamable_http import streamablehttp_client
    return McpUpstream(name, lambda: streamablehttp_client(url, headers=headers))
