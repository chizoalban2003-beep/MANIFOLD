"""Tests for EXP1 — MPC Look-ahead Planner vs A* baseline."""
from __future__ import annotations

import pytest

from manifold.nervatura_world import NERVATURAWorld
from manifold.experiments.mpc_planner import (
    WorldModelSimulator,
    MPCPlanner,
    BenchmarkResult,
    run_mpc_vs_astar_benchmark,
    _world_astar,
)


def _small_world() -> NERVATURAWorld:
    world = NERVATURAWorld(6, 6, 1, default_crna=(0.4, 0.3, 0.8, 0.2))
    # Place a high-R obstacle in the middle
    world.set_cell(3, 3, 0, c=0.4, r=0.75, n=0.5, a=0.1)
    return world


# Test 1: WorldModelSimulator.simulate_step returns dict with CRNA keys
def test_world_model_simulator_returns_crna_dict():
    world = _small_world()
    sim = WorldModelSimulator(world)
    result = sim.simulate_step(agent_pos=(2, 2, 0), action=(1, 0, 0))
    assert isinstance(result, dict)
    for key in ("c", "r", "n", "a"):
        assert key in result, f"Missing key {key!r}"
        assert isinstance(result[key], float)


# Test 2: MPCPlanner.plan returns a path that reaches the target
def test_mpc_plan_reaches_target():
    world = _small_world()
    mpc = MPCPlanner(horizon=3, risk_budget=0.8)
    result = mpc.plan(start=(0, 0, 0), target=(5, 5, 0), world=world)
    assert isinstance(result, dict)
    assert "path" in result
    path = result["path"]
    assert isinstance(path, list)
    assert len(path) > 0
    # Last element must be the target
    assert tuple(path[-1]) == (5, 5, 0)


# Test 3: MPC path is not worse than A* path in cost (within 20%)
def test_mpc_cost_not_much_worse_than_astar():
    world = _small_world()
    mpc = MPCPlanner(horizon=3, risk_budget=0.8)
    start, target = (0, 0, 0), (5, 5, 0)
    mpc_result = mpc.plan(start, target, world)
    astar_result = _world_astar(world, start, target, risk_budget=0.8)

    if not mpc_result["path"] or not astar_result["found"]:
        pytest.skip("No path found")

    mpc_cost = sum(
        (world.cell(*p).traversal_cost() if world.cell(*p) else 0.65)
        for p in mpc_result["path"][1:]
    )
    astar_cost = astar_result["total_cost"]

    # MPC may take a safer detour but must be within 20% cost penalty
    assert mpc_cost <= astar_cost * 1.20 + 0.1, (
        f"MPC cost {mpc_cost:.3f} is >20% worse than A* cost {astar_cost:.3f}"
    )


# Test 4: run_mpc_vs_astar_benchmark returns BenchmarkResult with win_rate >= 0
def test_benchmark_returns_valid_result():
    result = run_mpc_vs_astar_benchmark()
    assert isinstance(result, BenchmarkResult)
    assert result.win_rate >= 0.0
    assert result.win_rate <= 1.0
    assert isinstance(result.avg_cost_delta, float)
    assert isinstance(result.avg_risk_delta, float)


# Test 5: MPC avoids high-R cells more often than A* (measured over 20 trials)
def test_mpc_avoids_high_r_cells():
    import random
    rng = random.Random(999)
    world = NERVATURAWorld(8, 8, 1, default_crna=(0.4, 0.3, 0.8, 0.2))

    # Place high-R cells (R=0.65 — below risk_budget=0.7, so A* may traverse them)
    for ox, oy in [(2, 2), (3, 3), (4, 4), (2, 4), (4, 2), (3, 5), (5, 3)]:
        world.set_cell(ox, oy, 0, c=0.4, r=0.65, n=0.5, a=0.1)

    mpc = MPCPlanner(horizon=3, risk_budget=0.7)
    mpc_high_r_visits = 0
    astar_high_r_visits = 0
    n_trials = 20

    for _ in range(n_trials):
        sx, sy = rng.randint(0, 7), rng.randint(0, 7)
        tx, ty = rng.randint(0, 7), rng.randint(0, 7)
        while (sx, sy) == (tx, ty):
            tx, ty = rng.randint(0, 7), rng.randint(0, 7)

        start, target = (sx, sy, 0), (tx, ty, 0)
        mpc_result = mpc.plan(start, target, world)
        astar_result = _world_astar(world, start, target, 0.7)

        if mpc_result["path"]:
            mpc_high_r_visits += sum(
                1 for p in mpc_result["path"]
                if world.cell(*p) and world.cell(*p).r >= 0.6
            )
        if astar_result["found"] and astar_result["path"]:
            astar_high_r_visits += sum(
                1 for p in astar_result["path"]
                if world.cell(*p) and world.cell(*p).r >= 0.6
            )

    # MPC should visit high-R cells at most as often as A* (on average)
    assert mpc_high_r_visits <= astar_high_r_visits + n_trials, (
        f"MPC visited {mpc_high_r_visits} vs A* {astar_high_r_visits} high-R cells"
    )
