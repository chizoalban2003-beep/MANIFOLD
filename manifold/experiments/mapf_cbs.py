"""EXP3 — CBS Multi-Agent Pathfinding (MAPF).

Tests whether Conflict-Based Search finds collision-free paths for
multiple agents simultaneously, vs the current ATS right-of-way
arbitration.  No new dependencies.

Fix (8.2): Added edge-conflict detection — agents swapping positions in
one timestep.  CBS now branches on both vertex and swap conflicts, and
the low-level A* accepts edge constraints in addition to vertex constraints.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass
from typing import Optional

from manifold.nervatura_world import NERVATURAWorld


# ---------------------------------------------------------------------------
# Shared time-expanded A* for CBS low-level solver
# ---------------------------------------------------------------------------

def _neighbors(pos: tuple) -> list:
    x, y, z = pos
    result = []
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nb = (x + dx, y + dy, z + dz)
        if nb[0] >= 0 and nb[1] >= 0 and nb[2] >= 0:
            result.append(nb)
    return result


def _world_astar_timed(
    world: NERVATURAWorld,
    start: tuple,
    target: tuple,
    risk_budget: float = 0.7,
    forbidden_times: Optional[set] = None,
    forbidden_edges: Optional[set] = None,
    max_timestep: int = 60,
) -> list:
    """Time-expanded A* that avoids vertex and edge constraint pairs.

    forbidden_times:  set of (cell, timestep) — vertex constraints.
    forbidden_edges:  set of ((from_cell, to_cell), timestep) — edge constraints.

    Allows 'wait-in-place' actions so CBS can always resolve conflicts.
    Returns a list of positions (repeated = waits), or [] if infeasible.
    """
    forbidden = forbidden_times or set()
    forbidden_e = forbidden_edges or set()
    start = tuple(start)
    target = tuple(target)

    def h(pos: tuple) -> float:
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) + abs(pos[2] - target[2])

    def cell_cv_rv(pos: tuple):
        c = world.cell(*pos)
        return (c.c, c.r) if c is not None else (0.5, 0.5)

    open_heap: list = []
    heapq.heappush(open_heap, (h(start), 0.0, 0, start))
    g_scores: dict = {(start, 0): 0.0}
    came_from: dict = {}

    while open_heap:
        f, g, t, pos = heapq.heappop(open_heap)
        state = (pos, t)

        if g > g_scores.get(state, float("inf")):
            continue

        if pos == target:
            path: list = []
            cur = state
            while cur in came_from:
                path.append(cur[0])
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path

        if t >= max_timestep:
            continue

        next_t = t + 1

        for nb in _neighbors(pos):
            if (nb, next_t) in forbidden:
                continue
            if ((pos, nb), next_t) in forbidden_e:
                continue
            cv, rv = cell_cv_rv(nb)
            if rv > risk_budget:
                continue
            new_g = g + cv + rv * 0.3
            new_state = (nb, next_t)
            if new_g < g_scores.get(new_state, float("inf")):
                g_scores[new_state] = new_g
                came_from[new_state] = state
                heapq.heappush(open_heap, (new_g + h(nb), new_g, next_t, nb))

        # Wait in place
        if (pos, next_t) not in forbidden:
            new_g = g + 0.05
            new_state = (pos, next_t)
            if new_g < g_scores.get(new_state, float("inf")):
                g_scores[new_state] = new_g
                came_from[new_state] = state
                heapq.heappush(open_heap, (new_g + h(pos), new_g, next_t, pos))

    return []


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """A constraint forbidding an agent from a vertex or edge at a timestep.

    If edge_from is set, this is an edge constraint: agent must not traverse
    edge (edge_from → cell) at timestep.  Otherwise vertex constraint.
    """
    agent_id: str
    cell: tuple
    timestep: int
    edge_from: Optional[tuple] = None


@dataclass
class Conflict:
    """A conflict between two agents.

    conflict_type: "vertex" or "edge"
    For edge conflicts cell_b holds the second cell of the swap.
    """
    agent_a: str
    agent_b: str
    cell: tuple
    timestep: int
    conflict_type: str = "vertex"
    cell_b: Optional[tuple] = None


@dataclass
class CBSNode:
    """A node in the CBS search tree."""
    constraints: list
    paths: dict
    cost: float


_CBS_COUNTER = itertools.count()


# ---------------------------------------------------------------------------
# CBSSolver
# ---------------------------------------------------------------------------

class CBSSolver:
    """Conflict-Based Search (CBS) multi-agent pathfinder.

    Detects both vertex conflicts (same cell at same time) and edge
    conflicts (two agents swapping positions in one timestep).
    """

    def solve(
        self,
        agents: list,
        world: NERVATURAWorld,
        risk_budget: float = 0.7,
        max_expansions: int = 200,
    ) -> dict:
        """Find conflict-free paths for all agents.

        Parameters
        ----------
        agents:
            List of dicts: ``{id, start, target}``
        world:
            NERVATURAWorld the agents navigate.
        risk_budget:
            Maximum R value for traversable cells.
        max_expansions:
            Safety cap on CBS tree expansions.

        Returns
        -------
        dict with keys:
            paths, total_cost, conflicts_resolved, feasible, expansions_used
        """
        initial_paths: dict = {}
        for agent in agents:
            path = _world_astar_timed(
                world,
                tuple(agent["start"]),
                tuple(agent["target"]),
                risk_budget,
            )
            initial_paths[agent["id"]] = path

        root = CBSNode(
            constraints=[],
            paths=initial_paths,
            cost=float(sum(len(p) for p in initial_paths.values())),
        )

        open_list: list = []
        heapq.heappush(open_list, (root.cost, next(_CBS_COUNTER), root))
        expansions = 0

        while open_list and expansions < max_expansions:
            _, _, node = heapq.heappop(open_list)
            expansions += 1

            conflict = self.find_first_conflict(node.paths)
            if conflict is None:
                return {
                    "paths": node.paths,
                    "total_cost": node.cost,
                    "conflicts_resolved": expansions - 1,
                    "feasible": True,
                    "expansions_used": expansions,
                }

            for constrained_id in [conflict.agent_a, conflict.agent_b]:
                if conflict.conflict_type == "edge":
                    # Each agent is forbidden the edge it was traversing during the swap
                    if constrained_id == conflict.agent_a:
                        new_constraint = Constraint(
                            agent_id=constrained_id,
                            cell=conflict.cell_b,
                            timestep=conflict.timestep,
                            edge_from=conflict.cell,
                        )
                    else:
                        new_constraint = Constraint(
                            agent_id=constrained_id,
                            cell=conflict.cell,
                            timestep=conflict.timestep,
                            edge_from=conflict.cell_b,
                        )
                else:
                    new_constraint = Constraint(
                        agent_id=constrained_id,
                        cell=conflict.cell,
                        timestep=conflict.timestep,
                    )

                child_constraints = list(node.constraints) + [new_constraint]
                agent_data = next(
                    (a for a in agents if a["id"] == constrained_id), None
                )
                if agent_data is None:
                    continue

                new_path = self.replan_with_constraint(
                    constrained_id,
                    child_constraints,
                    world,
                    agent_data["start"],
                    agent_data["target"],
                    risk_budget,
                )
                child_paths = dict(node.paths)
                child_paths[constrained_id] = new_path
                child_cost = float(sum(len(p) for p in child_paths.values()))

                child_node = CBSNode(
                    constraints=child_constraints,
                    paths=child_paths,
                    cost=child_cost,
                )
                heapq.heappush(open_list, (child_cost, next(_CBS_COUNTER), child_node))

        best_node = node if open_list or expansions > 0 else root
        return {
            "paths": best_node.paths,
            "total_cost": best_node.cost,
            "conflicts_resolved": expansions,
            "feasible": self.find_first_conflict(best_node.paths) is None,
            "expansions_used": expansions,
        }

    def find_first_conflict(self, paths: dict) -> Optional[Conflict]:
        """Return the first vertex or edge conflict, or None.

        Vertex conflict: two agents at the same cell at the same timestep.
        Edge conflict: two agents swap positions between t-1 and t
                       (A: X→Y while B: Y→X simultaneously).
        """
        agent_ids = list(paths.keys())
        if len(agent_ids) < 2:
            return None

        max_len = max((len(p) for p in paths.values()), default=0)

        # Vertex conflicts
        for t in range(max_len):
            positions_at_t: dict = {}
            for aid in agent_ids:
                path = paths[aid]
                if not path:
                    continue
                pos = tuple(path[t] if t < len(path) else path[-1])
                if pos in positions_at_t:
                    return Conflict(
                        agent_a=positions_at_t[pos],
                        agent_b=aid,
                        cell=pos,
                        timestep=t,
                        conflict_type="vertex",
                    )
                positions_at_t[pos] = aid

        # Edge (swap) conflicts
        for t in range(1, max_len):
            for i, aid_a in enumerate(agent_ids):
                for aid_b in agent_ids[i + 1:]:
                    path_a = paths[aid_a]
                    path_b = paths[aid_b]
                    if not path_a or not path_b:
                        continue
                    pa_prev = tuple(path_a[t - 1] if t - 1 < len(path_a) else path_a[-1])
                    pa_curr = tuple(path_a[t] if t < len(path_a) else path_a[-1])
                    pb_prev = tuple(path_b[t - 1] if t - 1 < len(path_b) else path_b[-1])
                    pb_curr = tuple(path_b[t] if t < len(path_b) else path_b[-1])

                    # Swap: A moved from X→Y while B moved from Y→X
                    if pa_prev == pb_curr and pb_prev == pa_curr and pa_prev != pb_prev:
                        return Conflict(
                            agent_a=aid_a,
                            agent_b=aid_b,
                            cell=pa_prev,
                            timestep=t,
                            conflict_type="edge",
                            cell_b=pb_prev,
                        )

        return None

    def replan_with_constraint(
        self,
        agent_id: str,
        constraints: list,
        world: NERVATURAWorld,
        start,
        target,
        risk_budget: float = 0.7,
    ) -> list:
        """Re-run time-expanded A* with all constraints for *agent_id* active."""
        forbidden_times: set = set()
        forbidden_edges: set = set()
        for c in constraints:
            if c.agent_id == agent_id:
                if c.edge_from is not None:
                    forbidden_edges.add(((c.edge_from, c.cell), c.timestep))
                else:
                    forbidden_times.add((c.cell, c.timestep))
        return _world_astar_timed(
            world,
            tuple(start),
            tuple(target),
            risk_budget,
            forbidden_times,
            forbidden_edges,
        )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_cbs_vs_rightofway_benchmark() -> dict:
    """Compare CBS vs ATS right-of-way over 20 crossing scenarios.

    Returns
    -------
    dict with keys:
        cbs_conflict_rate, rightofway_conflict_rate,
        cbs_total_cost, rightofway_total_cost,
        cbs_completion_rate, rightofway_completion_rate,
        edge_conflicts_encountered
    """
    rng = random.Random(2024)
    world = NERVATURAWorld(10, 10, 1, default_crna=(0.4, 0.3, 0.8, 0.3))
    solver = CBSSolver()

    cbs_conflicts = 0
    row_conflicts = 0
    cbs_total = 0.0
    row_total = 0.0
    cbs_completed = 0
    row_completed = 0
    edge_conflicts_encountered = 0
    n_scenarios = 20

    for scenario in range(n_scenarios):
        agents = []
        used_positions: set = set()
        for i in range(4):
            while True:
                sx, sy = rng.randint(0, 9), rng.randint(0, 9)
                tx, ty = rng.randint(0, 9), rng.randint(0, 9)
                if (
                    (sx, sy, 0) not in used_positions
                    and (tx, ty, 0) not in used_positions
                    and (sx, sy) != (tx, ty)
                ):
                    used_positions.add((sx, sy, 0))
                    used_positions.add((tx, ty, 0))
                    agents.append({
                        "id": f"agent-{scenario}-{i}",
                        "start": (sx, sy, 0),
                        "target": (tx, ty, 0),
                    })
                    break

        cbs_result = solver.solve(agents, world, risk_budget=0.8)
        remaining = solver.find_first_conflict(cbs_result["paths"])
        if remaining is not None:
            cbs_conflicts += 1
            if remaining.conflict_type == "edge":
                edge_conflicts_encountered += 1
        cbs_total += cbs_result["total_cost"]
        if all(cbs_result["paths"].get(a["id"]) for a in agents):
            cbs_completed += 1

        row_paths: dict = {}
        for idx, agent in enumerate(agents):
            path = _world_astar_timed(world, agent["start"], agent["target"], 0.8)
            row_paths[agent["id"]] = [agent["start"]] * idx + path if path else []
        row_conflict = solver.find_first_conflict(row_paths)
        if row_conflict is not None:
            row_conflicts += 1
        row_total += sum(len(p) for p in row_paths.values())
        if all(row_paths.get(a["id"]) for a in agents):
            row_completed += 1

    return {
        "cbs_conflict_rate": round(cbs_conflicts / n_scenarios, 4),
        "rightofway_conflict_rate": round(row_conflicts / n_scenarios, 4),
        "cbs_total_cost": round(cbs_total, 4),
        "rightofway_total_cost": round(row_total, 4),
        "cbs_completion_rate": round(cbs_completed / n_scenarios, 4),
        "rightofway_completion_rate": round(row_completed / n_scenarios, 4),
        "edge_conflicts_encountered": edge_conflicts_encountered,
    }
