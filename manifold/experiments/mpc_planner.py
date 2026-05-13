"""EXP1 — MPC Look-ahead Planner vs A* baseline.

Tests whether N-step look-ahead planning improves path quality
over the current single-step CRNAPlanner.  No new dependencies.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Optional

from manifold.nervatura_world import NERVATURAWorld


# ---------------------------------------------------------------------------
# Shared A* on NERVATURAWorld (no global DynamicGrid dependency)
# ---------------------------------------------------------------------------

def _world_astar(
    world: NERVATURAWorld,
    start: tuple,
    target: tuple,
    risk_budget: float = 0.7,
    forbidden_cells: Optional[set] = None,
    max_steps: int = 2000,
) -> dict:
    """A* path planner operating directly on a NERVATURAWorld instance."""
    forbidden = forbidden_cells or set()
    start = tuple(start)
    target = tuple(target)

    def h(pos: tuple) -> float:
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) + abs(pos[2] - target[2])

    def cell_cv_rv(pos: tuple):
        c = world.cell(*pos)
        return (c.c, c.r) if c is not None else (0.5, 0.5)

    open_heap: list = []
    heapq.heappush(open_heap, (h(start), 0.0, 0.0, start))
    g_scores: dict = {start: 0.0}
    came_from: dict = {}
    steps = 0

    while open_heap and steps < max_steps:
        _f, g, risk, pos = heapq.heappop(open_heap)
        steps += 1

        if pos == target:
            path: list = []
            cur = pos
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return {"found": True, "path": path, "total_cost": g, "total_risk": risk}

        if g > g_scores.get(pos, float("inf")):
            continue

        x, y, z = pos
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nb = (x + dx, y + dy, z + dz)
            if nb[0] < 0 or nb[1] < 0 or nb[2] < 0:
                continue
            if nb in forbidden:
                continue
            cv, rv = cell_cv_rv(nb)
            if rv > risk_budget:
                continue
            new_g = g + cv + rv * 0.3
            if new_g < g_scores.get(nb, float("inf")):
                g_scores[nb] = new_g
                came_from[nb] = pos
                new_risk = risk + rv
                heapq.heappush(open_heap, (new_g + h(nb), new_g, new_risk, nb))

    return {"found": False, "path": [], "total_cost": 0.0, "total_risk": 0.0}


# ---------------------------------------------------------------------------
# WorldModelSimulator
# ---------------------------------------------------------------------------

class WorldModelSimulator:
    """Predict the next CRNA state when an agent moves into a cell."""

    def __init__(self, world: NERVATURAWorld) -> None:
        self._world = world

    def simulate_step(self, agent_pos: tuple, action: tuple) -> dict:
        """Return predicted CRNA state dict after agent takes *action* from *agent_pos*.

        Model rules
        -----------
        * Moving into a cell reduces N by 0.1 (exploration reduces neutrality).
        * If the cell has R > 0.6, there is a ~20% chance of an obstacle event
          (R spikes further).  Deterministic: uses hash of target cell position.
        """
        dx, dy, dz = action
        nx = agent_pos[0] + dx
        ny = agent_pos[1] + dy
        nz = agent_pos[2] + dz

        cell = self._world.cell(nx, ny, nz)
        if cell is None:
            return {"c": 0.5, "r": 0.5, "n": 1.0, "a": 0.0}

        predicted_n = max(0.0, cell.n - 0.1)
        predicted_c = cell.c
        predicted_a = cell.a
        predicted_r = cell.r

        # Deterministic ~20% obstacle event for high-R cells
        if cell.r > 0.6:
            if abs(hash((nx, ny, nz))) % 10 < 2:
                predicted_r = min(1.0, cell.r + 0.2)

        return {
            "c": round(predicted_c, 4),
            "r": round(predicted_r, 4),
            "n": round(predicted_n, 4),
            "a": round(predicted_a, 4),
        }


# ---------------------------------------------------------------------------
# MPCPlanner
# ---------------------------------------------------------------------------

class MPCPlanner:
    """N-step look-ahead MPC planner.  Generates candidate paths, scores them
    using simulated horizon states, and returns the lowest-score path.
    """

    def __init__(self, horizon: int = 3, risk_budget: float = 0.7) -> None:
        self.horizon = horizon
        self.risk_budget = risk_budget

    def plan(
        self,
        start: tuple,
        target: tuple,
        world: NERVATURAWorld,
    ) -> dict:
        """Return the best path with MPC scoring.

        Returns
        -------
        dict with keys:
            path, score, horizon_used, simulated_states, vs_astar_cost_delta
        """
        start = tuple(start)
        target = tuple(target)
        sim = WorldModelSimulator(world)

        # Baseline A* path
        astar_result = _world_astar(world, start, target, self.risk_budget)
        astar_cost = astar_result["total_cost"]

        # Candidate paths: baseline + one path starting via each neighbour
        candidates: list[list] = []
        if astar_result["found"] and astar_result["path"]:
            candidates.append(astar_result["path"])

        x, y, z = start
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            waypoint = (x + dx, y + dy, z + dz)
            if waypoint[0] < 0 or waypoint[1] < 0 or waypoint[2] < 0:
                continue
            wc = world.cell(*waypoint)
            if wc is None or wc.r > self.risk_budget:
                continue
            sub = _world_astar(world, waypoint, target, self.risk_budget)
            if sub["found"] and sub["path"]:
                candidates.append([start] + sub["path"])

        if not candidates:
            return {
                "path": [],
                "score": float("inf"),
                "horizon_used": self.horizon,
                "simulated_states": [],
                "vs_astar_cost_delta": 0.0,
            }

        best_path: list = []
        best_score = float("inf")
        best_simulated: list = []

        for path in candidates:
            if len(path) < 2:
                continue
            simulated_states: list = []
            pos = path[0]
            for step_idx in range(min(self.horizon, len(path) - 1)):
                next_pos = path[step_idx + 1]
                action = (
                    next_pos[0] - pos[0],
                    next_pos[1] - pos[1],
                    next_pos[2] - pos[2],
                )
                state = sim.simulate_step(pos, action)
                simulated_states.append(state)
                pos = next_pos

            if not simulated_states:
                continue

            n_steps = len(simulated_states)
            exp_cost = sum(s["c"] for s in simulated_states) / n_steps
            exp_risk = sum(s["r"] for s in simulated_states) / n_steps
            exp_asset = sum(s["a"] for s in simulated_states) / n_steps

            # MPC scoring: cost * (1 + risk) - asset
            score = exp_cost * (1.0 + exp_risk) - exp_asset

            if score < best_score:
                best_score = score
                best_path = path
                best_simulated = simulated_states

        if not best_path:
            # Fallback to A* result
            best_path = astar_result["path"]
            best_simulated = []
            best_score = float("inf")

        # Compute actual traversal cost of best_path on world
        mpc_cost = sum(
            (world.cell(*p).traversal_cost() if world.cell(*p) else 0.65)
            for p in best_path[1:]
        )

        return {
            "path": best_path,
            "score": round(best_score, 6),
            "horizon_used": self.horizon,
            "simulated_states": best_simulated,
            "vs_astar_cost_delta": round(mpc_cost - astar_cost, 6),
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results from comparing MPC vs A* over many trials."""
    win_rate: float          # fraction of trials where MPC score <= A* score
    avg_cost_delta: float    # mean(mpc_cost - astar_cost); negative = MPC cheaper
    avg_risk_delta: float    # mean(mpc_risk - astar_risk); negative = MPC safer


def run_mpc_vs_astar_benchmark() -> BenchmarkResult:
    """Compare MPCPlanner vs CRNAPlanner over 50 random start→target pairs.

    Creates a 10×10×1 NERVATURAWorld with 15 medium-risk obstacle cells
    (R=0.65, below risk_budget=0.7 so A* traverses them but MPC scores
    them more harshly due to the (1+risk) multiplier).

    Returns
    -------
    BenchmarkResult with win_rate, avg_cost_delta, avg_risk_delta.
    """
    rng = random.Random(2024)

    world = NERVATURAWorld(10, 10, 1, default_crna=(0.4, 0.3, 0.8, 0.3))

    # Place 15 medium-risk obstacle cells (R=0.65 — traversable but penalised)
    obstacle_positions = set()
    attempts = 0
    while len(obstacle_positions) < 15 and attempts < 100:
        ox = rng.randint(1, 8)
        oy = rng.randint(1, 8)
        obstacle_positions.add((ox, oy, 0))
        attempts += 1
    for ox, oy, oz in obstacle_positions:
        world.set_cell(ox, oy, oz, c=0.4, r=0.65, n=0.5, a=0.2)

    mpc = MPCPlanner(horizon=3, risk_budget=0.7)

    wins = 0
    total_cost_delta = 0.0
    total_risk_delta = 0.0
    trials = 50

    for _ in range(trials):
        sx, sy = rng.randint(0, 9), rng.randint(0, 9)
        tx, ty = rng.randint(0, 9), rng.randint(0, 9)
        while (tx, ty) == (sx, sy):
            tx, ty = rng.randint(0, 9), rng.randint(0, 9)

        start = (sx, sy, 0)
        target = (tx, ty, 0)

        mpc_result = mpc.plan(start, target, world)
        astar_result = _world_astar(world, start, target, 0.7)

        if not mpc_result["path"] or not astar_result["found"]:
            continue

        # Measure risk encountered (count of high-R cell visits)
        mpc_risk = sum(
            (world.cell(*p).r if world.cell(*p) else 0.5)
            for p in mpc_result["path"]
        )
        astar_risk = sum(
            (world.cell(*p).r if world.cell(*p) else 0.5)
            for p in astar_result["path"]
        )

        mpc_cost = sum(
            (world.cell(*p).traversal_cost() if world.cell(*p) else 0.65)
            for p in mpc_result["path"][1:]
        )
        astar_cost = astar_result["total_cost"]

        total_cost_delta += mpc_cost - astar_cost
        total_risk_delta += mpc_risk - astar_risk

        # MPC "wins" if its risk-weighted score is <= A*'s
        mpc_score = mpc_cost * (1.0 + mpc_risk / max(len(mpc_result["path"]), 1))
        astar_score = astar_cost * (1.0 + astar_risk / max(len(astar_result["path"]), 1))
        if mpc_score <= astar_score:
            wins += 1

    n = max(trials, 1)
    return BenchmarkResult(
        win_rate=round(wins / n, 4),
        avg_cost_delta=round(total_cost_delta / n, 6),
        avg_risk_delta=round(total_risk_delta / n, 6),
    )
