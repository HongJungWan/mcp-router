"""Real MCP upstreams over the official SDK (stdio / streamable-HTTP), with
health-checking and automatic reconnect.

Each upstream owns a dedicated asyncio loop in a background thread. `_serve()` is
a supervisor loop: connect -> open ClientSession -> initialize -> park while
periodically pinging the session; if a ping fails (the subprocess/transport
died) or the context exits, it reconnects with exponential backoff. `list_tools`
/`call_tool` submit coroutines onto that loop and block for the result; concurrent
calls are safe — they run on the one loop and the SDK multiplexes them by request
id (no extra lock needed).

The session factory is injectable so the reconnect state machine is unit-tested
without the SDK; the default builds a real `mcp.ClientSession` (lazy import,
pip install .[mcp]). While an upstream is down, calls raise UpstreamError until it
reconnects (per-attempt backoff is bounded, but total downtime is unbounded with
the default max_retries=0). connect/initialize and the health ping are each
timeout-bounded so a soft-dead server can't hang the loop. `health()` reports
liveness + reconnect count.
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
            self._loop.call_soon_threadsafe(fut.cancel)
            raise

    def spawn(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def call_soon(self, fn) -> None:
        self._loop.call_soon_threadsafe(fn)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()


def _text(result) -> str:
    parts = [getattr(c, "text", "") for c in (result.content or []) if getattr(c, "type", "") == "text"]
    return "\n".join(p for p in parts if p)


def _default_session_factory(streams):
    from mcp import ClientSession                       # lazy: needs .[mcp]
    return ClientSession(streams[0], streams[1])         # stdio=(r,w); http=(r,w,get_session_id)


class McpUpstream:
    """An open MCP ClientSession behind the sync Upstream interface, with reconnect."""

    def __init__(self, name: str, connect_factory: Callable,
                 session_factory: Callable = _default_session_factory,
                 init_timeout: float = 30.0, health_interval: float = 15.0,
                 health_timeout: float = 10.0, backoff_base: float = 0.5,
                 backoff_cap: float = 10.0, max_retries: int = 0):
        self.name = name
        self._connect = connect_factory
        self._session_factory = session_factory
        self._init_timeout = init_timeout
        self._health_interval = health_interval
        self._health_timeout = health_timeout
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._max_retries = max_retries              # 0 => retry forever
        self._lt = _LoopThread()
        self._session = None
        self._healthy = False
        self._reconnects = 0
        self._err: Optional[BaseException] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._closed = False
        self._fut = self._lt.spawn(self._serve())
        if not self._ready.wait(timeout=init_timeout) or self._session is None:
            self.close()
            raise UpstreamError(f"mcp upstream '{name}' failed to initialize: {self._err}")

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        attempt = 0
        while not self._stop.is_set():
            try:
                async with self._connect() as streams:
                    async with self._session_factory(streams) as session:
                        await asyncio.wait_for(session.initialize(), timeout=self._init_timeout)
                        self._session = session
                        self._healthy = True
                        self._err = None
                        attempt = 0
                        self._ready.set()
                        await self._park(session)        # returns on stop OR ping failure
            except Exception as e:
                self._err = e
            finally:
                self._session = None
                self._healthy = False
            if self._stop.is_set():
                break
            attempt += 1
            self._reconnects += 1
            if self._max_retries and attempt > self._max_retries:
                break
            await self._backoff(attempt)                 # cancellable by stop
        self._ready.set()                                # never leave __init__ blocked

    async def _park(self, session) -> None:
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._health_interval)
                return                                   # stop requested
            except asyncio.TimeoutError:
                pass
            try:
                await asyncio.wait_for(session.send_ping(), timeout=self._health_timeout)
            except Exception as e:                       # raised OR timed out (soft death)
                self._err = e
                return                                   # session dead -> reconnect

    async def _backoff(self, attempt: int) -> None:
        delay = min(self._backoff_cap, self._backoff_base * (2 ** (attempt - 1)))
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    def _require(self):
        s = self._session
        if s is None or not self._healthy:
            raise UpstreamError(f"upstream '{self.name}' session is not available (reconnecting?)")
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

    def health(self) -> dict:
        return {"name": self.name, "healthy": self._healthy, "reconnects": self._reconnects,
                "last_error": None if self._err is None else str(self._err)}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stop is not None:
            self._lt.call_soon(self._stop.set)      # graceful: let _serve unwind its contexts
        try:
            self._fut.result(timeout=5)
        except FuturesTimeout:                       # hung outside a stop-aware await -> cancel
            self._lt.call_soon(self._fut.cancel)
            try:
                self._fut.result(timeout=5)
            except Exception:
                pass
        except Exception:
            pass
        self._lt.stop()


def stdio_upstream(name: str, command: str, args: Optional[list] = None,
                   env: Optional[dict] = None, cwd: Optional[str] = None, **kw) -> McpUpstream:
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=command, args=args or [], env=env, cwd=cwd)
    return McpUpstream(name, lambda: stdio_client(params), **kw)


def http_upstream(name: str, url: str, headers: Optional[dict] = None, **kw) -> McpUpstream:
    from mcp.client.streamable_http import streamablehttp_client
    return McpUpstream(name, lambda: streamablehttp_client(url, headers=headers), **kw)
