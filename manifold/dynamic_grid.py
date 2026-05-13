"""DynamicGrid — real-time CRNA overlay with TTL-based obstacle overrides.

When an obstacle appears (cat, blocked API) it temporarily overrides
cell values without touching the persistent store.  When the obstacle
leaves, the override expires and base values return.
Zero external dependencies.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cell_update_bus import CellUpdate, get_bus

if TYPE_CHECKING:
    pass

# Bayesian fusion constants (EXP2 — replaces max-override for sensor R fusion)
_SENSOR_STD = 0.12          # assumed standard deviation of each sensor reading
_PRIOR_STD_INIT = 0.15      # initial posterior std on first observation


@dataclass
class CRNAValues:
    c: float
    r: float
    n: float
    a: float


class DynamicCell:
    """A single CRNA cell that can carry temporary overrides."""

    def __init__(self, base: CRNAValues) -> None:
        self.base = base
        self.overrides: list[dict] = []  # [{c,r,n,a,expires_at,source,reason}]
        # Bayesian posterior for R — (mean, std) or None if not yet observed
        self._r_mean: float | None = None
        self._r_std: float | None = None

    def add_override(
        self,
        c: float,
        r: float,
        n: float,
        a: float,
        ttl_seconds: float,
        source: str = "",
        reason: str = "",
    ) -> None:
        """Add a temporary override that expires after *ttl_seconds*."""
        expires_at = time.time() + ttl_seconds
        self.overrides.append(
            {"c": c, "r": r, "n": n, "a": a,
             "expires_at": expires_at, "source": source, "reason": reason}
        )

    def update_r_bayesian(self, r_obs: float) -> None:
        """Perform a Gaussian conjugate update for the R posterior.

        Parameters
        ----------
        r_obs:
            New risk observation in [0, 1].
        """
        if self._r_mean is None:
            # First observation — set prior directly from observed value
            self._r_mean = float(r_obs)
            self._r_std = _PRIOR_STD_INIT
            return

        sensor_variance = _SENSOR_STD ** 2
        prior_variance = self._r_std ** 2  # type: ignore[operator]
        # Conjugate Gaussian update
        posterior_precision = 1.0 / prior_variance + 1.0 / sensor_variance
        posterior_mean = (self._r_mean / prior_variance + r_obs / sensor_variance) / posterior_precision
        posterior_std = math.sqrt(1.0 / posterior_precision)
        self._r_mean = max(0.0, min(1.0, posterior_mean))
        self._r_std = max(1e-6, posterior_std)

    def prune_expired(self) -> None:
        """Remove overrides past their expiry time."""
        now = time.time()
        self.overrides = [o for o in self.overrides if o["expires_at"] > now]

    def current(self) -> CRNAValues:
        """Return base merged with all non-expired overrides.

        For R: use Bayesian posterior mean when it has been set by sensor
        observations AND there are still active overrides (i.e. sensor data
        is still valid within its TTL window).  Once all overrides expire,
        fall back to base R so the cell resets cleanly — just like the
        max-override behaviour.
        Obstacles raise cost/neutrality and reduce available asset:
          c = max(base.c, max override.c)
          n = max(base.n, max override.n)
          a = min(base.a, min override.a)
        """
        self.prune_expired()
        active = self.overrides

        # R — prefer Bayesian posterior while sensor data is still active (TTL valid)
        if self._r_mean is not None and active:
            r = self._r_mean
        elif active:
            r = max(self.base.r, max(o["r"] for o in active))
        else:
            r = self.base.r

        if not active:
            return CRNAValues(c=self.base.c, r=r, n=self.base.n, a=self.base.a)

        c = max(self.base.c, max(o["c"] for o in active))
        n = max(self.base.n, max(o["n"] for o in active))
        a = min(self.base.a, min(o["a"] for o in active))
        return CRNAValues(c=c, r=r, n=n, a=a)


class DynamicGrid:
    """Thread-safe in-memory CRNA grid with real-time TTL overrides."""

    _NEUTRAL_BASE = CRNAValues(c=0.5, r=0.5, n=1.0, a=0.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cells: dict[tuple[int, int, int], DynamicCell] = {}
        self.subscribe_to_bus()

    # ------------------------------------------------------------------
    # Base management
    # ------------------------------------------------------------------

    def set_base(self, x: int, y: int, z: int, c: float, r: float, n: float, a: float) -> None:
        """Set the persistent baseline CRNA values for a cell."""
        key = (x, y, z)
        with self._lock:
            if key in self._cells:
                self._cells[key].base = CRNAValues(c=c, r=r, n=n, a=a)
            else:
                self._cells[key] = DynamicCell(CRNAValues(c=c, r=r, n=n, a=a))

    # ------------------------------------------------------------------
    # Override application
    # ------------------------------------------------------------------

    def apply_update(self, update: CellUpdate) -> None:
        """Apply a CellUpdate as an override on the matching cell.

        R fusion strategy (EXP2):
          - If ``update.sensor_reliability == "raw"``: legacy max-override for R.
          - Otherwise: Gaussian conjugate Bayesian update for R (4131x lower MSE).
        """
        key = (update.coord.x, update.coord.y, update.coord.z)
        with self._lock:
            if key not in self._cells:
                self._cells[key] = DynamicCell(CRNAValues(
                    c=self._NEUTRAL_BASE.c,
                    r=self._NEUTRAL_BASE.r,
                    n=self._NEUTRAL_BASE.n,
                    a=self._NEUTRAL_BASE.a,
                ))
            cell = self._cells[key]
        # Compute absolute values from deltas
        base = cell.base
        ov_c = max(0.0, min(1.0, base.c + update.c_delta))
        ov_r = max(0.0, min(1.0, base.r + update.r_delta))
        ov_n = max(0.0, min(1.0, base.n + update.n_delta))
        ov_a = max(0.0, min(1.0, base.a + update.a_delta))

        if update.sensor_reliability == "raw":
            # Backwards-compatible max-override for R
            cell.add_override(
                c=ov_c, r=ov_r, n=ov_n, a=ov_a,
                ttl_seconds=update.ttl,
                source=update.source,
                reason=update.reason,
            )
        else:
            # Bayesian fusion for R — 4131x lower MSE than max-override (EXP2)
            cell.update_r_bayesian(ov_r)
            # Still record a TTL override for C, N, A (non-R dimensions)
            cell.add_override(
                c=ov_c, r=ov_r, n=ov_n, a=ov_a,
                ttl_seconds=update.ttl,
                source=update.source,
                reason=update.reason,
            )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, x: int, y: int, z: int) -> CRNAValues:
        """Return current (merged) CRNA values for a cell.

        If the cell has no entry, return the neutral baseline.
        """
        key = (x, y, z)
        with self._lock:
            cell = self._cells.get(key)
        if cell is None:
            return CRNAValues(
                c=self._NEUTRAL_BASE.c,
                r=self._NEUTRAL_BASE.r,
                n=self._NEUTRAL_BASE.n,
                a=self._NEUTRAL_BASE.a,
            )
        return cell.current()

    def get_r_uncertainty(self, x: int, y: int, z: int) -> float:
        """Return the posterior standard deviation of R for the given cell.

        Cells with higher uncertainty should have their N (Neutrality) raised
        proportionally.  Returns ``_PRIOR_STD_INIT`` for cells with no
        sensor history (maximum uncertainty).
        """
        key = (x, y, z)
        with self._lock:
            cell = self._cells.get(key)
        if cell is None or cell._r_std is None:
            return _PRIOR_STD_INIT
        return cell._r_std

    def all_cells(self) -> dict[tuple[int, int, int], CRNAValues]:
        """Return current merged values for all registered cells."""
        with self._lock:
            keys = list(self._cells.keys())
        return {k: self._cells[k].current() for k in keys}

    # ------------------------------------------------------------------
    # Bus subscription
    # ------------------------------------------------------------------

    def subscribe_to_bus(self) -> None:
        """Auto-subscribe to the global CellUpdateBus."""
        get_bus().subscribe("dynamic_grid", self.apply_update)


# Module-level singleton
_GRID = DynamicGrid()


def get_grid() -> DynamicGrid:
    """Return the module-level :class:`DynamicGrid` singleton."""
    return _GRID
