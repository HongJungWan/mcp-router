"""Circuit breaker: closed -> open -> half-open -> closed.

Guards a flaky upstream so one failing server trips fast instead of stalling
every request. The clock is injected (`now`) so tests are deterministic and the
open->half-open cooldown transition is exercised without sleeping.

Thread-safe: the gateway serves via ThreadingHTTPServer, so all state
transitions are guarded by a lock and half-open admits exactly one probe (others
fast-fail) to avoid a thundering herd against a still-down upstream. The upstream
call itself runs OUTSIDE the lock so healthy calls stay concurrent.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is open (or a probe is already in flight)."""


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    now: Callable[[], float] = time.monotonic

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _probing: bool = field(default=False, init=False)
    trips: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _effective(self) -> BreakerState:
        # caller holds the lock. OPEN auto-promotes to HALF_OPEN once cooldown elapses.
        if self._state is BreakerState.OPEN and self.now() - self._opened_at >= self.reset_timeout:
            return BreakerState.HALF_OPEN
        return self._state

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._effective()

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            st = self._effective()
            if st is BreakerState.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' is open")
            if st is BreakerState.HALF_OPEN:
                if self._probing:                       # another thread owns the probe
                    raise CircuitOpenError(f"circuit '{self.name}' is half-open (probe in flight)")
                self._probing = True
                self._state = BreakerState.HALF_OPEN
        # run the upstream call without holding the lock
        try:
            result = fn()
        except Exception:
            with self._lock:
                self._on_failure()
            raise
        with self._lock:
            self._on_success()
        return result

    def _on_success(self) -> None:
        self._probing = False
        if self._state is BreakerState.HALF_OPEN:       # probe passed -> close
            self._state = BreakerState.CLOSED
        if self._state is BreakerState.CLOSED:
            self._failures = 0
        # if concurrently tripped to OPEN by others, a late success does NOT reopen it

    def _on_failure(self) -> None:
        if self._probing:                               # probe failed -> straight back to open
            self._probing = False
            self._trip()
            return
        if self._state is BreakerState.OPEN:
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self.now()
        self.trips += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"name": self.name, "state": self._effective().value,
                    "failures": self._failures, "trips": self.trips}
