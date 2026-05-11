"""DigitalHealthMonitor — live API/tool health → CRNA cell updates.

Monitors registered digital tools and APIs.  When a tool degrades
(high error rate, rate-limited, slow) it publishes a CellUpdate via
CellUpdateBus to raise R/C in the relevant grid cells.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .cell_update_bus import CellCoord, CellUpdate, get_bus

_log = logging.getLogger(__name__)


@dataclass
class ToolHealth:
    tool_id: str
    endpoint_url: str = ""
    grid_coord: tuple = (0, 0, 3)  # default data layer z=3
    check_interval: int = 10       # seconds between health checks
    last_status: str = "unknown"   # healthy | degraded | down | rate_limited
    error_count: int = 0
    success_count: int = 0
    last_latency_ms: float = 0.0
    last_checked: float = field(default_factory=time.time)

    def error_rate(self) -> float:
        total = self.error_count + self.success_count
        return self.error_count / total if total > 0 else 0.0


class DigitalHealthMonitor:
    """Thread-safe monitor that converts tool health events into CRNA updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tools: dict[str, ToolHealth] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        tool_id: str,
        endpoint_url: str = "",
        grid_coord: tuple = (0, 0, 3),
        check_interval: int = 10,
    ) -> ToolHealth:
        """Register a tool for health monitoring."""
        th = ToolHealth(
            tool_id=tool_id,
            endpoint_url=endpoint_url,
            grid_coord=grid_coord,
            check_interval=check_interval,
        )
        with self._lock:
            self._tools[tool_id] = th
        return th

    def _get_or_create(self, tool_id: str) -> ToolHealth:
        with self._lock:
            if tool_id not in self._tools:
                self._tools[tool_id] = ToolHealth(tool_id=tool_id)
            return self._tools[tool_id]

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(self, tool_id: str, success: bool, latency_ms: float = 0.0) -> None:
        """Called by pipeline when a tool call completes."""
        th = self._get_or_create(tool_id)
        with self._lock:
            if success:
                th.success_count += 1
                th.last_status = "healthy"
            else:
                th.error_count += 1
            th.last_latency_ms = latency_ms
            th.last_checked = time.time()
            er = th.error_rate()
            coord = th.grid_coord

        if er > 0.8:
            self._publish(coord, r_delta=0.7, c_delta=0.5, ttl=30.0, source=tool_id, reason="tool_degraded")
            with self._lock:
                th.last_status = "down"
        elif er > 0.5:
            self._publish(coord, r_delta=0.4, c_delta=0.0, ttl=30.0, source=tool_id, reason="high_error_rate")
            with self._lock:
                th.last_status = "degraded"

    def record_rate_limit(self, tool_id: str, retry_after_seconds: float = 60.0) -> None:
        """Called when a 429 is received from a tool."""
        th = self._get_or_create(tool_id)
        with self._lock:
            th.last_status = "rate_limited"
            coord = th.grid_coord
        self._publish(coord, r_delta=0.3, c_delta=0.8, ttl=retry_after_seconds,
                      source=tool_id, reason="rate_limited")

    def record_recovery(self, tool_id: str) -> None:
        """Called when errors stop — signals recovery."""
        th = self._get_or_create(tool_id)
        with self._lock:
            th.last_status = "healthy"
            coord = th.grid_coord
        self._publish(coord, r_delta=-0.4, c_delta=0.0, ttl=5.0,
                      source=tool_id, reason="recovery")

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background monitoring daemon thread."""
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._monitor_loop, daemon=True, name="manifold-health-monitor")
        t.start()

    def _monitor_loop(self) -> None:
        while self._running:
            time.sleep(5)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return summary of all tool health statuses."""
        with self._lock:
            return {
                tid: {
                    "tool_id": th.tool_id,
                    "status": th.last_status,
                    "error_rate": th.error_rate(),
                    "error_count": th.error_count,
                    "success_count": th.success_count,
                    "last_latency_ms": th.last_latency_ms,
                    "last_checked": th.last_checked,
                    "grid_coord": list(th.grid_coord),
                }
                for tid, th in self._tools.items()
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(
        self,
        coord: tuple,
        *,
        r_delta: float = 0.0,
        c_delta: float = 0.0,
        ttl: float = 30.0,
        source: str = "",
        reason: str = "",
    ) -> None:
        x, y, z = (int(v) for v in coord[:3])
        update = CellUpdate(
            coord=CellCoord(x=x, y=y, z=z),
            c_delta=c_delta,
            r_delta=r_delta,
            source=source,
            ttl=ttl,
            reason=reason,
        )
        get_bus().publish(update)
