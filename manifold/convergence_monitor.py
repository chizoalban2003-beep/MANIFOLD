"""manifold/convergence_monitor.py — Real-time NERVATURA convergence tracker.

EXP5 result: V(t) = Σ|CRNA_t - CRNA_mean|² decreased 39.5% over 500 steps
(11.72 → 7.09).  70.7% of steps were monotonically decreasing.  Stabilisation
at step ~394.  Decay rate -0.00101/step.

This confirms the theoretical NERVATURA claim: the system self-organises toward
a stable equilibrium without central coordination.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.nervatura_world import NERVATURAWorld


@dataclass
class ConvergenceSnapshot:
    """One V(t) measurement point."""

    timestamp: float
    v_lyapunov: float      # Σ|CRNA_t - CRNA_mean|² across all world cells
    delta_v: float         # V(t) - V(t-1), or 0.0 for the first snapshot
    is_converging: bool    # True when delta_v < 0
    mean_n: float          # mean Neutrality across the grid
    active_cells: int      # number of cells in the world


class ConvergenceMonitor:
    """Real-time NERVATURA stability tracker.

    Maintains a rolling window of :class:`ConvergenceSnapshot` objects and
    diagnoses whether the Lyapunov function V(t) is decreasing (converging),
    stable, or increasing (diverging).
    """

    def __init__(self, world: "NERVATURAWorld", window: int = 50) -> None:
        self._world = world
        self._window = window
        self._history: deque[ConvergenceSnapshot] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Core measurement
    # ------------------------------------------------------------------

    def snapshot(self) -> ConvergenceSnapshot:
        """Compute V(t) for all cells and append to the rolling window.

        V(t) = Σ |CRNA_t - CRNA_mean|²  averaged over the 4 CRNA dimensions.
        """
        cells = list(self._world._cells.values())
        if not cells:
            snap = ConvergenceSnapshot(
                timestamp=time.time(),
                v_lyapunov=0.0,
                delta_v=0.0,
                is_converging=False,
                mean_n=1.0,
                active_cells=0,
            )
            with self._lock:
                self._history.append(snap)
            return snap

        n = len(cells)
        mean_c = sum(c.c for c in cells) / n
        mean_r = sum(c.r for c in cells) / n
        mean_n_val = sum(c.n for c in cells) / n
        mean_a = sum(c.a for c in cells) / n

        v = sum(
            (c.c - mean_c) ** 2
            + (c.r - mean_r) ** 2
            + (c.n - mean_n_val) ** 2
            + (c.a - mean_a) ** 2
            for c in cells
        ) / n

        with self._lock:
            prev_v = self._history[-1].v_lyapunov if self._history else v
            delta_v = v - prev_v
            snap = ConvergenceSnapshot(
                timestamp=time.time(),
                v_lyapunov=round(v, 6),
                delta_v=round(delta_v, 6),
                is_converging=delta_v < 0,
                mean_n=round(mean_n_val, 4),
                active_cells=n,
            )
            self._history.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Health assessment
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True when the last 20 snapshots show V is not consistently increasing.

        Specifically: True when the fraction of snapshots with ``delta_v <= 0``
        (converging or plateau) is >= 0.5.  A ratio < 0.5 means governance
        dynamics are diverging — alert needed.
        """
        with self._lock:
            recent = list(self._history)[-20:]
        if not recent:
            return True  # no data yet — assume healthy
        non_diverging = sum(1 for s in recent if s.delta_v <= 0)
        return non_diverging / len(recent) >= 0.5

    def convergence_report(self) -> dict:
        """Return a structured convergence status report."""
        with self._lock:
            history = list(self._history)

        if not history:
            return {
                "v_current": None,
                "v_trend": None,
                "monotone_ratio_recent": None,
                "estimated_steps_to_stable": None,
                "health": "unknown",
                "recommendation": "No snapshots recorded yet. Call snapshot() first.",
            }

        v_current = history[-1].v_lyapunov
        recent_20 = history[-20:]
        deltas = [s.delta_v for s in recent_20]
        v_trend = sum(deltas) / len(deltas) if deltas else 0.0
        monotone_ratio = sum(1 for s in recent_20 if s.delta_v <= 0) / len(recent_20)

        # Estimate steps to stable (V stops decreasing)
        if v_trend < 0 and abs(v_trend) > 1e-9:
            estimated_steps = max(0, int(v_current / abs(v_trend)))
        else:
            estimated_steps = 0

        if monotone_ratio >= 0.7:
            health = "converging"
            recommendation = "V(t) is decreasing. System is self-organising normally."
        elif monotone_ratio >= 0.5:
            health = "stable"
            recommendation = "V(t) has plateaued. Monitor for divergence."
        else:
            health = "diverging"
            recommendation = (
                "WARNING: V(t) is increasing. Governance may be destabilising. "
                "Consider pausing high-risk agents and reviewing recent policy changes."
            )

        return {
            "v_current": v_current,
            "v_trend": round(v_trend, 6),
            "monotone_ratio_recent": round(monotone_ratio, 4),
            "estimated_steps_to_stable": estimated_steps,
            "health": health,
            "recommendation": recommendation,
            "snapshots_collected": len(history),
        }

    # ------------------------------------------------------------------
    # Background daemon
    # ------------------------------------------------------------------

    def start(self, interval_seconds: float = 30.0) -> None:
        """Start a background daemon thread that calls snapshot() periodically.

        If is_healthy() returns False, publishes a ``convergence_warning``
        event on the CellUpdateBus so the world dashboard can alert.
        """
        if self._running:
            return
        self._running = True

        def _loop() -> None:
            while self._running:
                try:
                    self.snapshot()
                    if not self.is_healthy():
                        self._publish_warning()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(interval_seconds)

        self._thread = threading.Thread(
            target=_loop, daemon=True, name="manifold-convergence-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background daemon thread."""
        self._running = False

    def _publish_warning(self) -> None:
        """Publish a convergence_warning event to the CellUpdateBus."""
        try:
            from manifold.cell_update_bus import get_bus, CellUpdate, CellCoord  # noqa: PLC0415
            bus = get_bus()
            bus.publish(CellUpdate(
                coord=CellCoord(x=0, y=0, z=0),
                source="convergence_monitor",
                reason="convergence_warning",
            ))
        except Exception:  # noqa: BLE001
            pass
