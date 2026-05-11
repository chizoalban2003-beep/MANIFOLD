"""Tests for manifold.dynamic_grid."""
import time

import pytest

from manifold.cell_update_bus import CellCoord, CellUpdate
from manifold.dynamic_grid import CRNAValues, DynamicCell, DynamicGrid


@pytest.fixture()
def grid():
    """Fresh DynamicGrid for each test (not the singleton)."""
    g = DynamicGrid.__new__(DynamicGrid)
    import threading
    g._lock = threading.Lock()
    g._cells = {}
    # Don't subscribe to bus to keep tests isolated
    return g


def test_set_base_then_get_returns_base(grid):
    grid.set_base(1, 2, 0, c=0.3, r=0.6, n=0.2, a=0.8)
    v = grid.get(1, 2, 0)
    assert abs(v.c - 0.3) < 1e-9
    assert abs(v.r - 0.6) < 1e-9
    assert abs(v.n - 0.2) < 1e-9
    assert abs(v.a - 0.8) < 1e-9


def test_add_override_raises_r_above_base(grid):
    grid.set_base(0, 0, 0, c=0.2, r=0.1, n=0.5, a=0.9)
    grid._cells[(0, 0, 0)].add_override(c=0.2, r=0.85, n=0.5, a=0.9, ttl_seconds=60.0)
    v = grid.get(0, 0, 0)
    assert v.r == pytest.approx(0.85)


def test_expired_override_is_ignored(grid):
    grid.set_base(2, 2, 0, c=0.2, r=0.1, n=0.3, a=0.5)
    # TTL of 0 — expires immediately
    grid._cells[(2, 2, 0)].add_override(c=0.9, r=0.9, n=0.9, a=0.0, ttl_seconds=0.0)
    time.sleep(0.01)
    v = grid.get(2, 2, 0)
    # Expired — base values should return
    assert v.r == pytest.approx(0.1)


def test_base_restored_after_override_expires(grid):
    grid.set_base(3, 3, 0, c=0.2, r=0.1, n=0.3, a=0.5)
    grid._cells[(3, 3, 0)].add_override(c=0.9, r=0.9, n=0.9, a=0.0, ttl_seconds=0.05)
    v_before = grid.get(3, 3, 0)
    assert v_before.r == pytest.approx(0.9)
    time.sleep(0.1)
    v_after = grid.get(3, 3, 0)
    assert v_after.r == pytest.approx(0.1)


def test_apply_update_triggers_override(grid):
    grid.set_base(5, 5, 0, c=0.2, r=0.1, n=0.3, a=0.5)
    update = CellUpdate(
        coord=CellCoord(x=5, y=5, z=0),
        r_delta=0.8,
        ttl=60.0,
        source="sensor",
        reason="cat",
    )
    grid.apply_update(update)
    v = grid.get(5, 5, 0)
    assert v.r > 0.1  # override applied


def test_all_cells_returns_merged_values(grid):
    grid.set_base(0, 0, 0, c=0.1, r=0.2, n=0.3, a=0.4)
    grid.set_base(1, 0, 0, c=0.5, r=0.6, n=0.7, a=0.8)
    all_v = grid.all_cells()
    assert (0, 0, 0) in all_v
    assert (1, 0, 0) in all_v
    assert abs(all_v[(0, 0, 0)].c - 0.1) < 1e-9
    assert abs(all_v[(1, 0, 0)].r - 0.6) < 1e-9
