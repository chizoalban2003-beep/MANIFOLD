"""DynamicGrid — real-time CRNA overlay with TTL-based obstacle overrides.

When an obstacle appears (cat, blocked API) it temporarily overrides
cell values without touching the persistent store.  When the obstacle
leaves, the override expires and base values return.
Zero external dependencies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cell_update_bus import CellUpdate, get_bus

if TYPE_CHECKING:
    pass


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

    def prune_expired(self) -> None:
        """Remove overrides past their expiry time."""
        now = time.time()
        self.overrides = [o for o in self.overrides if o["expires_at"] > now]

    def current(self) -> CRNAValues:
        """Return base merged with all non-expired overrides.

        Obstacles raise risk/cost/neutrality and reduce available asset:
          r = max(base.r, max override.r)
          c = max(base.c, max override.c)
          n = max(base.n, max override.n)
          a = min(base.a, min override.a)
        """
        self.prune_expired()
        active = self.overrides
        if not active:
            return CRNAValues(c=self.base.c, r=self.base.r, n=self.base.n, a=self.base.a)
        r = max(self.base.r, max(o["r"] for o in active))
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
        """Apply a CellUpdate as an override on the matching cell."""
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
