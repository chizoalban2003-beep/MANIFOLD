"""EXP-A — Bayesian CRNA + CBS Integration.

Wires BayesianGrid as the cell-value source for CBSSolver, so agents
plan collision-free paths over uncertainty-quantified terrain.

Research question: does CBS produce better (lower-cost, lower-risk) paths
when the grid is more thoroughly observed (lower entropy)?  We measure
plan quality as a function of the grid's entropy_map().

No new dependencies.
"""

from __future__ import annotations

import random
from typing import Any

from manifold.experiments.bayesian_crna import BayesianGrid
from manifold.experiments.mapf_cbs import CBSSolver
from manifold.nervatura_world import NERVATURAWorld


# ---------------------------------------------------------------------------
# BayesianWorldAdapter
# ---------------------------------------------------------------------------

class BayesianWorldAdapter:
    """Wraps BayesianGrid + NERVATURAWorld so CBSSolver can use Bayesian CRNA.

    CBSSolver calls world.cell(x, y, z) for cost and risk values.
    This adapter overrides those values with Bayesian posterior means,
    leaving the NERVATURAWorld skeleton for navigation structure.
    """

    def __init__(self, base_world: NERVATURAWorld, bayes_grid: BayesianGrid) -> None:
        self._world = base_world
        self._bayes = bayes_grid
        # Expose _cells for _world_astar_timed neighbor checks
        self._cells = base_world._cells

    def cell(self, x: int, y: int, z: int):
        """Return a cell-like object using Bayesian posterior means."""
        scalars = self._bayes.get_scalar(x, y, z)
        base = self._world.cell(x, y, z)
        if base is None:
            return None
        # Shadow the base cell's c and r with Bayesian estimates
        base.c = scalars["c"]
        base.r = scalars["r"]
        base.n = scalars["n"]
        base.a = scalars["a"]
        return base


def _simulate_sensor_campaign(
    grid: BayesianGrid,
    width: int,
    depth: int,
    n_observations: int,
    rng: random.Random,
    sensor_reliability: float = 0.9,
) -> None:
    """Simulate sensors updating grid cells with noisy readings."""
    noise_std = (1.0 - sensor_reliability) ** 0.5
    for _ in range(n_observations):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, depth - 1)
        reading = {
            "c": max(0.0, min(1.0, 0.4 + rng.gauss(0, noise_std))),
            "r": max(0.0, min(1.0, 0.3 + rng.gauss(0, noise_std))),
            "n": max(0.0, min(1.0, 0.5 + rng.gauss(0, noise_std))),
            "a": max(0.0, min(1.0, 0.3 + rng.gauss(0, noise_std))),
        }
        grid.update_from_sensor(x, y, 0, reading, sensor_reliability)


def _path_quality(paths: dict, world: BayesianWorldAdapter) -> dict[str, float]:
    """Compute aggregate cost, risk, and entropy along all agent paths."""
    total_cost = 0.0
    total_risk = 0.0
    total_entropy = 0.0
    step_count = 0
    entropy_map = world._bayes.entropy_map()

    for path in paths.values():
        for pos in path:
            pos = tuple(pos)
            scalars = world._bayes.get_scalar(*pos)
            total_cost += scalars["c"]
            total_risk += scalars["r"]
            total_entropy += entropy_map.get(pos, 0.0)
            step_count += 1

    if step_count == 0:
        return {"avg_cost": 0.0, "avg_risk": 0.0, "avg_path_entropy": 0.0}
    return {
        "avg_cost": round(total_cost / step_count, 4),
        "avg_risk": round(total_risk / step_count, 4),
        "avg_path_entropy": round(total_entropy / step_count, 4),
    }


def run_bayesian_cbs_benchmark() -> dict[str, Any]:
    """Compare CBS path quality on low-entropy vs high-entropy grids.

    Protocol:
    - Two 10×10 grids, same topology.
    - High-entropy grid: 0 sensor updates (maximum uncertainty).
    - Low-entropy grid: 200 sensor updates (well-observed terrain).
    - CBS plans 4-agent paths on both.
    - Metrics: avg path cost, avg path risk, avg path entropy.

    Returns
    -------
    dict with keys: high_entropy_metrics, low_entropy_metrics,
        entropy_reduction, quality_improvement
    """
    rng = random.Random(99)
    width, depth = 10, 10
    base_crna = (0.4, 0.3, 1.0, 0.3)

    solver = CBSSolver()

    agents = [
        {"id": "a0", "start": (0, 0, 0), "target": (9, 9, 0)},
        {"id": "a1", "start": (9, 0, 0), "target": (0, 9, 0)},
        {"id": "a2", "start": (0, 5, 0), "target": (9, 5, 0)},
        {"id": "a3", "start": (5, 0, 0), "target": (5, 9, 0)},
    ]

    results = {}

    for label, n_observations in [("high_entropy", 0), ("low_entropy", 200)]:
        base_world = NERVATURAWorld(width, depth, 1, default_crna=base_crna)
        bayes_grid = BayesianGrid()

        if n_observations > 0:
            _simulate_sensor_campaign(
                bayes_grid, width, depth, n_observations, rng, sensor_reliability=0.9
            )

        adapter = BayesianWorldAdapter(base_world, bayes_grid)

        cbs_result = solver.solve(agents, adapter, risk_budget=0.8)
        quality = _path_quality(cbs_result["paths"], adapter)

        # Mean entropy across all cells in the grid
        entropy_map = bayes_grid.entropy_map()
        all_cell_keys = [(x, y, 0) for x in range(width) for y in range(depth)]
        mean_entropy = (
            sum(entropy_map.get(k, 0.0) for k in all_cell_keys) / len(all_cell_keys)
        )

        results[label] = {
            **quality,
            "mean_grid_entropy": round(mean_entropy, 6),
            "feasible": cbs_result["feasible"],
            "total_cost": round(cbs_result["total_cost"], 4),
        }

    # Quality improvement: lower risk and cost on low-entropy grid = positive
    he = results["high_entropy"]
    le = results["low_entropy"]
    entropy_reduction = round(he["mean_grid_entropy"] - le["mean_grid_entropy"], 6)
    risk_improvement = round(he["avg_risk"] - le["avg_risk"], 4)
    cost_improvement = round(he["avg_cost"] - le["avg_cost"], 4)

    return {
        "high_entropy_metrics": results["high_entropy"],
        "low_entropy_metrics": results["low_entropy"],
        "entropy_reduction": entropy_reduction,
        "risk_improvement": risk_improvement,
        "cost_improvement": cost_improvement,
        "hypothesis_confirmed": risk_improvement > 0 or cost_improvement > 0,
    }
