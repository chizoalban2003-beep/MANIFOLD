"""Tests for manifold.nervatura_world (NERVATURAWorld)."""
import json

import pytest

from manifold.nervatura_world import NERVATURAWorld, NERVATURACell


def test_world_initialises_with_correct_cell_count():
    world = NERVATURAWorld(width=4, depth=4, height=2)
    assert world.summary()["total_cells"] == 4 * 4 * 2


def test_set_cell_updates_crna_values():
    world = NERVATURAWorld(width=5, depth=5, height=1)
    world.set_cell(2, 2, 0, c=0.3, r=0.7, n=0.1, a=0.9, domain="kitchen")
    cell = world.cell(2, 2, 0)
    assert cell is not None
    assert cell.c == pytest.approx(0.3)
    assert cell.r == pytest.approx(0.7)
    assert cell.domain == "kitchen"


def test_neighbours_returns_up_to_6_adjacent_cells():
    world = NERVATURAWorld(width=5, depth=5, height=5)
    nbs = world.neighbours(2, 2, 2)
    assert len(nbs) == 6
    # Corner cell (0,0,0) has only 3 neighbours
    nbs_corner = world.neighbours(0, 0, 0)
    assert len(nbs_corner) == 3


def test_reduce_neutrality_lowers_n():
    world = NERVATURAWorld(width=3, depth=3, height=1)
    cell = world.cell(1, 1, 0)
    assert cell.n == pytest.approx(1.0)
    cell.reduce_neutrality(0.4)
    assert cell.n == pytest.approx(0.6)
    assert cell.age == 1


def test_terraform_lowers_c():
    world = NERVATURAWorld(width=3, depth=3, height=1)
    cell = world.cell(1, 1, 0)
    original_c = cell.c
    cell.terraform(cost_reduction=0.2)
    assert cell.c < original_c


def test_harvest_reduces_a_and_returns_amount():
    world = NERVATURAWorld(width=3, depth=3, height=1, default_crna=(0.5, 0.5, 1.0, 1.0))
    cell = world.cell(1, 1, 0)
    assert cell.a == pytest.approx(1.0)
    got = cell.harvest(0.3)
    assert got == pytest.approx(0.3)
    assert cell.a == pytest.approx(0.7)


def test_diffuse_neutrality_raises_n_over_time():
    world = NERVATURAWorld(width=3, depth=3, height=1)
    # Lower n first
    for cell in world._cells.values():
        cell.n = 0.5
    world.diffuse_neutrality(decay=0.1)
    for cell in world._cells.values():
        assert cell.n == pytest.approx(0.6)


def test_to_json_from_json_round_trip():
    world = NERVATURAWorld(width=3, depth=3, height=2)
    world.set_cell(1, 1, 0, c=0.2, r=0.8, n=0.3, a=0.7, domain="test")
    serialised = world.to_json()
    restored = NERVATURAWorld.from_json(serialised)
    assert restored.width == 3
    assert restored.depth == 3
    cell = restored.cell(1, 1, 0)
    assert cell is not None
    assert cell.c == pytest.approx(0.2)
    assert cell.r == pytest.approx(0.8)
    assert cell.domain == "test"
