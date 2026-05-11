"""Integration tests — real-time obstacle handling (v1.7.0)."""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifold.cell_update_bus import CellCoord, CellUpdate, CellUpdateBus
from manifold.dynamic_grid import DynamicGrid
from manifold.planner import CRNAPlanner
from manifold.health_monitor import DigitalHealthMonitor


# ---------------------------------------------------------------------------
# Isolated fixture: fresh bus + grid + planner for every test
# ---------------------------------------------------------------------------

@pytest.fixture()
def rt_env(monkeypatch):
    """Fresh real-time environment: bus, grid, planner, monitor."""
    bus = CellUpdateBus()
    grid = DynamicGrid.__new__(DynamicGrid)
    grid._lock = threading.Lock()
    grid._cells = {}
    # Wire grid to test bus
    grid.subscribe_to_bus = lambda: bus.subscribe("dynamic_grid", grid.apply_update)
    grid.subscribe_to_bus()

    monitor = DigitalHealthMonitor()

    # Monkeypatch all three singletons
    import manifold.cell_update_bus as bus_mod
    import manifold.dynamic_grid as dg_mod
    import manifold.health_monitor as hm_mod
    import manifold.planner as pl_mod

    monkeypatch.setattr(bus_mod, "get_bus", lambda: bus)
    monkeypatch.setattr(dg_mod, "get_grid", lambda: grid)
    monkeypatch.setattr(hm_mod, "get_bus", lambda: bus)
    monkeypatch.setattr(pl_mod, "get_grid", lambda: grid)

    planner = CRNAPlanner()

    yield bus, grid, planner, monitor


# ---------------------------------------------------------------------------
# Test 1: obstacle raises risk and blocks path
# ---------------------------------------------------------------------------

def test_obstacle_raises_risk_and_blocks_path(rt_env):
    bus, grid, planner, _monitor = rt_env

    # Set a clear cell at (5,5,0) with base R=0.1
    grid.set_base(5, 5, 0, c=0.1, r=0.1, n=0.3, a=0.5)

    # Publish a cat-detected update with r_delta=0.9
    bus.publish(CellUpdate(
        coord=CellCoord(x=5, y=5, z=0),
        r_delta=0.9,
        source="lidar-01",
        ttl=60.0,
        reason="cat detected",
    ))
    time.sleep(0.1)  # let bus deliver to grid

    # Verify the cell R is now elevated
    v = grid.get(5, 5, 0)
    assert v.r > 0.5, f"Expected R > 0.5 after cat update, got {v.r}"

    # Plan a path that would go through (5,5,0) — risk_budget=0.7
    # The cat raises R to ~1.0 so the cell is skipped
    result = planner.plan(start=(0, 0, 0), target=(10, 10, 0), risk_budget=0.7, max_steps=2000)
    # Path should not go through the obstacle cell
    assert (5, 5, 0) not in result["path"]


# ---------------------------------------------------------------------------
# Test 2: digital obstacle (rate limit) raises C in grid
# ---------------------------------------------------------------------------

def test_digital_obstacle_rate_limit_raises_c(rt_env):
    bus, grid, planner, monitor = rt_env

    grid.set_base(3, 3, 4, c=0.2, r=0.2, n=0.3, a=0.5)
    monitor.register_tool("tool1", grid_coord=(3, 3, 4))

    # Monkeypatch health_monitor to use our test bus
    import manifold.health_monitor as hm_mod
    original = hm_mod.get_bus
    hm_mod.get_bus = lambda: bus
    try:
        monitor.record_rate_limit("tool1", retry_after_seconds=60.0)
    finally:
        hm_mod.get_bus = original

    time.sleep(0.1)

    v = grid.get(3, 3, 4)
    # c should have risen (override applied)
    assert v.c > 0.2, f"Expected c > 0.2 after rate limit, got {v.c}"


# ---------------------------------------------------------------------------
# Test 3: obstacle clears (TTL expiry) and path restores
# ---------------------------------------------------------------------------

def test_obstacle_clears_and_path_restores(rt_env):
    bus, grid, planner, _monitor = rt_env

    # Set base R=0.1 for cell (4,4,0)
    grid.set_base(4, 4, 0, c=0.1, r=0.1, n=0.3, a=0.5)

    # Publish with very short TTL (0.1s)
    bus.publish(CellUpdate(
        coord=CellCoord(x=4, y=4, z=0),
        r_delta=0.9,
        source="sensor",
        ttl=0.1,
        reason="temporary obstacle",
    ))
    time.sleep(0.05)
    v_during = grid.get(4, 4, 0)
    assert v_during.r > 0.5, "Override should be active"

    # Wait for TTL to expire
    time.sleep(0.15)
    v_after = grid.get(4, 4, 0)
    # R should be back near base (0.1)
    assert v_after.r < 0.5, f"Expected R to return to base, got {v_after.r}"
