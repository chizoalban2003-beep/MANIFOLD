"""Tests for EXP3 — CBS Multi-Agent Pathfinding."""
from __future__ import annotations

from manifold.nervatura_world import NERVATURAWorld
from manifold.experiments.mapf_cbs import (
    CBSSolver,
    run_cbs_vs_rightofway_benchmark,
)


def _world() -> NERVATURAWorld:
    return NERVATURAWorld(8, 8, 1, default_crna=(0.4, 0.3, 0.8, 0.2))


# Test 1: CBSSolver.solve with 2 non-conflicting agents returns 2 paths
def test_cbs_two_non_conflicting_agents():
    world = _world()
    agents = [
        {"id": "a1", "start": (0, 0, 0), "target": (3, 0, 0)},
        {"id": "a2", "start": (0, 7, 0), "target": (3, 7, 0)},
    ]
    solver = CBSSolver()
    result = solver.solve(agents, world, risk_budget=0.8)

    assert "paths" in result
    assert len(result["paths"]) == 2
    assert "a1" in result["paths"]
    assert "a2" in result["paths"]
    assert len(result["paths"]["a1"]) > 0
    assert len(result["paths"]["a2"]) > 0


# Test 2: CBSSolver.solve with 2 conflicting agents resolves the conflict
def test_cbs_resolves_conflict():
    world = _world()
    # Two agents cross head-on along row y=4
    agents = [
        {"id": "left", "start": (0, 4, 0), "target": (7, 4, 0)},
        {"id": "right", "start": (7, 4, 0), "target": (0, 4, 0)},
    ]
    solver = CBSSolver()
    result = solver.solve(agents, world, risk_budget=0.9, max_expansions=500)

    assert result["feasible"], "CBS should find a feasible conflict-free solution"
    # The paths should not share the same cell at the same timestep
    conflict = solver.find_first_conflict(result["paths"])
    assert conflict is None, f"Unexpected conflict: {conflict}"


# Test 3: No two agents occupy the same cell at the same timestep in output
def test_no_conflicts_in_cbs_output():
    world = _world()
    agents = [
        {"id": "a0", "start": (0, 0, 0), "target": (6, 6, 0)},
        {"id": "a1", "start": (6, 0, 0), "target": (0, 6, 0)},
        {"id": "a2", "start": (0, 3, 0), "target": (6, 3, 0)},
    ]
    solver = CBSSolver()
    result = solver.solve(agents, world, risk_budget=0.9, max_expansions=500)

    conflict = solver.find_first_conflict(result["paths"])
    assert conflict is None, (
        f"Found unexpected conflict at cell={conflict.cell} t={conflict.timestep}"
        if conflict else ""
    )


# Test 4: run_cbs_vs_rightofway_benchmark returns conflict_rate < 0.5
def test_cbs_benchmark_low_conflict_rate():
    result = run_cbs_vs_rightofway_benchmark()
    assert "cbs_conflict_rate" in result
    assert result["cbs_conflict_rate"] < 0.5, (
        f"CBS conflict_rate {result['cbs_conflict_rate']} should be < 0.5"
    )


# Test 5: CBS total_cost <= right-of-way total_cost
def test_cbs_total_cost_le_rightofway():
    result = run_cbs_vs_rightofway_benchmark()
    assert "cbs_total_cost" in result
    assert "rightofway_total_cost" in result
    # CBS should be at most as expensive as right-of-way
    assert result["cbs_total_cost"] <= result["rightofway_total_cost"] + 1.0, (
        f"CBS cost {result['cbs_total_cost']} should not greatly exceed "
        f"right-of-way cost {result['rightofway_total_cost']}"
    )
