"""Tests for manifold.planner (CRNAPlanner)."""
import time

import pytest

from manifold.cell_update_bus import CellCoord, CellUpdate
from manifold.dynamic_grid import DynamicGrid
from manifold.planner import CRNAPlanner


@pytest.fixture(autouse=True)
def fresh_grid(monkeypatch):
    """Use a fresh DynamicGrid for each test."""
    import threading
    g = DynamicGrid.__new__(DynamicGrid)
    g._lock = threading.Lock()
    g._cells = {}
    import manifold.planner as p_mod
    monkeypatch.setattr(p_mod, "get_grid", lambda: g)
    return g


def test_plan_finds_direct_path(fresh_grid):
    planner = CRNAPlanner()
    result = planner.plan(start=(0, 0, 0), target=(3, 0, 0), risk_budget=0.9)
    assert result["found"] is True
    assert (0, 0, 0) in result["path"]
    assert (3, 0, 0) in result["path"]


def test_plan_routes_around_high_r_obstacle(fresh_grid):
    planner = CRNAPlanner()
    # Block (2,0,0) with high risk
    fresh_grid.set_base(2, 0, 0, c=0.1, r=0.99, n=0.1, a=0.0)
    result = planner.plan(start=(0, 0, 0), target=(4, 0, 0), risk_budget=0.7)
    assert result["found"] is True
    assert (2, 0, 0) not in result["path"]


def test_plan_returns_found_false_when_unreachable():
    """Target is completely surrounded by impassable cells."""
    import threading
    from manifold.dynamic_grid import DynamicGrid
    g = DynamicGrid.__new__(DynamicGrid)
    g._lock = threading.Lock()
    g._cells = {}
    # Surround (5,5,0) with walls
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(0,0,)]:
        x, y = 5 + (dx if len((dx, dy)) > 1 else 0), 5 + dy
        g.set_base(x, y, 0, c=1.0, r=1.0, n=0.0, a=0.0)
    g.set_base(5, 5, 0, c=1.0, r=1.0, n=0.0, a=0.0)

    planner = CRNAPlanner()
    import manifold.planner as p_mod
    original = p_mod.get_grid
    p_mod.get_grid = lambda: g
    try:
        result = planner.plan(start=(0, 0, 0), target=(5, 5, 0), risk_budget=0.01, max_steps=50)
        assert result["found"] is False
    finally:
        p_mod.get_grid = original


def test_risk_budget_zero_blocks_all_paths(fresh_grid):
    planner = CRNAPlanner()
    # With risk_budget=0, any cell with r>0 is skipped
    result = planner.plan(start=(0, 0, 0), target=(3, 3, 0), risk_budget=0.0, max_steps=100)
    # All cells have default r=0.5, so nothing is navigable → no path
    assert result["found"] is False


def test_replan_needed_returns_true_when_r_spikes(fresh_grid):
    planner = CRNAPlanner()
    path = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    fresh_grid.set_base(1, 0, 0, c=0.1, r=0.8, n=0.1, a=0.0)
    assert planner.replan_needed(path, threshold=0.5) is True


def test_neighbours_returns_6_face_adjacent(fresh_grid):
    planner = CRNAPlanner()
    nbs = planner.neighbours((5, 5, 5))
    assert len(nbs) == 6
    expected = {(6,5,5),(4,5,5),(5,6,5),(5,4,5),(5,5,6),(5,5,4)}
    assert set(nbs) == expected
