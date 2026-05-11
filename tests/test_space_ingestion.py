"""Tests for manifold_physical/space_ingestion.py."""
import os
import threading

import pytest

from manifold.dynamic_grid import DynamicGrid
from manifold_physical.space_ingestion import SpaceIngestion


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_floorplan.json")


@pytest.fixture(autouse=True)
def fresh_grid(monkeypatch):
    """Use a fresh DynamicGrid for each test to avoid cross-test state."""
    g = DynamicGrid.__new__(DynamicGrid)
    g._lock = threading.Lock()
    g._cells = {}
    import manifold.dynamic_grid as dg_mod
    monkeypatch.setattr(dg_mod, "get_grid", lambda: g)
    import manifold_physical.space_ingestion as si_mod
    monkeypatch.setattr(si_mod, "get_grid", lambda: g)
    return g


def test_load_floorplan_loads_valid_json():
    si = SpaceIngestion()
    fp = si.load_floorplan(FIXTURE_PATH)
    assert fp["name"] == "TestHome"
    assert "rooms" in fp
    assert len(fp["rooms"]) == 2


def test_ingest_populates_correct_number_of_cells(fresh_grid):
    si = SpaceIngestion()
    fp = si.load_floorplan(FIXTURE_PATH)
    count = si.ingest(fp)
    # Kitchen: 4*4*3=48, Baby Room: 3*3*3=27, Wall: 1*10*3=30, Corridor: 2*10*1=20 → 125
    assert count == 125


def test_wall_cells_have_c_1_0(fresh_grid):
    si = SpaceIngestion()
    fp = si.load_floorplan(FIXTURE_PATH)
    si.ingest(fp)
    # Wall is at x=4, y=[0,10), z=[0,3). Check z=1 which is not overwritten by corridor (z=[0,1))
    v = fresh_grid.get(4, 0, 1)
    assert v.c == pytest.approx(1.0)


def test_apply_room_policy_raises_r_for_named_room(fresh_grid):
    si = SpaceIngestion()
    fp = si.load_floorplan(FIXTURE_PATH)
    si.ingest(fp)
    # Apply a policy that raises r to 0.99 in Kitchen
    si.apply_room_policy("Kitchen", {"r_override": 0.99, "ttl": 3600, "reason": "night"}, fp)
    v = fresh_grid.get(1, 1, 0)
    assert v.r == pytest.approx(0.99)


def test_export_grid_summary_returns_per_room_averages(fresh_grid):
    si = SpaceIngestion()
    fp = si.load_floorplan(FIXTURE_PATH)
    si.ingest(fp)
    summary = si.export_grid_summary(fp)
    assert "rooms" in summary
    assert "Kitchen" in summary["rooms"]
    assert "avg_r" in summary["rooms"]["Kitchen"]
