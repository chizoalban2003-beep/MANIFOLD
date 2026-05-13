"""CellUpdateBus — real-time CRNA cell update publish/subscribe bus.

When a sensor detects an obstacle (physical or digital) it pushes a
CellUpdate event here.  Subscribers (agents, planner, world) receive it
immediately.  Thread-safe, zero external dependencies.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

_log = logging.getLogger(__name__)


@dataclass
class CellCoord:
    x: int
    y: int
    z: int = 0
    t: float = 0.0


@dataclass
class CellUpdate:
    coord: CellCoord
    c_delta: float = 0.0   # change to Cost
    r_delta: float = 0.0   # change to Risk
    n_delta: float = 0.0   # change to Neutrality
    a_delta: float = 0.0   # change to Asset
    source: str = ""       # who published this (sensor id, monitor id)
    ttl: float = 30.0      # seconds until this update expires
    timestamp: float = field(default_factory=time.time)
    reason: str = ""       # human-readable: 'cat detected', 'api_rate_limit'
    sensor_reliability: str = ""  # "raw" → max-override; else → Bayesian fusion


class CellUpdateBus:
    """Singleton pub/sub bus for real-time CRNA cell updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, Callable[[CellUpdate], None]] = {}
        self._ring: deque[CellUpdate] = deque(maxlen=500)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, subscriber_id: str, callback: Callable[[CellUpdate], None]) -> None:
        """Register *callback* to be called on every published update."""
        with self._lock:
            self._subscribers[subscriber_id] = callback

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a previously registered subscriber."""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, update: CellUpdate) -> None:
        """Push *update* to all subscribers (each in a background thread).

        The update is also appended to the internal ring buffer.
        """
        with self._lock:
            self._ring.append(update)
            callbacks = list(self._subscribers.values())

        for cb in callbacks:
            t = threading.Thread(target=self._safe_call, args=(cb, update), daemon=True)
            t.start()

    def _safe_call(self, cb: Callable[[CellUpdate], None], update: CellUpdate) -> None:
        try:
            cb(update)
        except Exception as exc:  # noqa: BLE001
            _log.debug("CellUpdateBus subscriber error: %s", exc)

    def publish_obstacle(
        self,
        x: int,
        y: int,
        z: int = 0,
        *,
        risk_spike: float = 0.9,
        source: str = "",
        ttl: float = 30.0,
        reason: str = "",
    ) -> None:
        """Convenience method: publish a risk-raising CellUpdate."""
        update = CellUpdate(
            coord=CellCoord(x=x, y=y, z=z),
            r_delta=risk_spike,
            source=source,
            ttl=ttl,
            reason=reason,
        )
        self.publish(update)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[CellUpdate]:
        """Return the most recent *limit* updates from the ring buffer."""
        with self._lock:
            items = list(self._ring)
        return items[-limit:]


# Module-level singleton
_BUS = CellUpdateBus()


def get_bus() -> CellUpdateBus:
    """Return the module-level :class:`CellUpdateBus` singleton."""
    return _BUS
