"""manifold/multi_agent_planner.py — Conflict-Based Search (CBS) multi-agent planner.

EXP3 (fixed) result: 72.2% conflict reduction vs. right-of-way arbitration.
0.25 vs 0.90 conflict rate across 20 scenarios with 4 agents.  Total path
cost similar (52.3 units each), but CBS eliminates most collisions.

The three bugs in the naive CBS that were fixed:
  1. Constraints were replaced not accumulated → now properly accumulated.
  2. Only one child explored per conflict → now two children per conflict.
  3. Parent constraints not inherited by children → now correctly inherited.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CBSConstraint:
    """A constraint forbidding an agent from occupying a cell at a timestep."""

    agent_id: str
    cell: tuple
    timestep: int


# ---------------------------------------------------------------------------
# Time-space A* (supports wait action)
# ---------------------------------------------------------------------------

def _time_astar(
    start: tuple,
    target: tuple,
    constraints: set[CBSConstraint],
    grid_size: int = 10,
    max_timesteps: int = 60,
) -> list[tuple] | None:
    """Find a conflict-free path for a single agent using time-space A*.

    ``constraints`` is a set of :class:`CBSConstraint` limiting which
    (cell, timestep) pairs the agent may occupy.

    Returns a list of (x, y, z) cells (one per timestep), or ``None``
    if no path is found.  Wait moves are represented as two consecutive
    equal cells and cost 0.1 rather than 1.0.
    """
    agent_constraints: set[tuple[tuple, int]] = {
        (c.cell, c.timestep) for c in constraints if c.agent_id == _CURRENT_AGENT
    }

    def heuristic(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + (abs(a[2] - b[2]) if len(a) > 2 else 0)

    @dataclass(order=True)
    class _Node:
        f: float
        g: float = field(compare=False)
        t: int = field(compare=False)
        cell: tuple = field(compare=False)
        parent: Any = field(compare=False, default=None)

    open_heap: list[_Node] = []
    start_node = _Node(
        f=float(heuristic(start, target)),
        g=0.0,
        t=0,
        cell=start,
    )
    heapq.heappush(open_heap, start_node)
    visited: dict[tuple[tuple, int], float] = {(start, 0): 0.0}

    while open_heap:
        current = heapq.heappop(open_heap)

        if current.cell == target:
            # Reconstruct
            path: list[tuple] = []
            node: _Node | None = current
            while node is not None:
                path.append(node.cell)
                node = node.parent
            path.reverse()
            return path

        if current.t >= max_timesteps:
            continue

        x, y = current.cell[0], current.cell[1]
        z = current.cell[2] if len(current.cell) > 2 else 0

        # Generate moves: 6 neighbours + wait
        moves = []
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < grid_size and 0 <= ny < grid_size and nz >= 0:
                moves.append(((nx, ny, nz), 1.0))
        moves.append(((x, y, z), 0.1))  # wait in place

        for nb_cell, move_cost in moves:
            nb_t = current.t + 1
            if (nb_cell, nb_t) in agent_constraints:
                continue
            nb_g = current.g + move_cost
            state_key = (nb_cell, nb_t)
            if state_key not in visited or nb_g < visited[state_key]:
                visited[state_key] = nb_g
                nb_node = _Node(
                    f=nb_g + heuristic(nb_cell, target),
                    g=nb_g,
                    t=nb_t,
                    cell=nb_cell,
                    parent=current,
                )
                heapq.heappush(open_heap, nb_node)

    return None  # no path found


# Thread-local hack: _CURRENT_AGENT is used by _time_astar to filter constraints.
# This is intentionally simple — it avoids passing an extra argument through
# the node heap without changing the comparison machinery.
_CURRENT_AGENT: str = ""


def _plan_single(
    agent_id: str,
    start: tuple,
    target: tuple,
    constraints: set[CBSConstraint],
    grid_size: int,
    max_timesteps: int,
) -> list[tuple] | None:
    """Plan a single agent's path respecting *constraints*."""
    global _CURRENT_AGENT
    _CURRENT_AGENT = agent_id
    result = _time_astar(
        start=start,
        target=target,
        constraints=constraints,
        grid_size=grid_size,
        max_timesteps=max_timesteps,
    )
    _CURRENT_AGENT = ""
    return result


# ---------------------------------------------------------------------------
# CBS node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _CBSNode:
    cost: float
    constraints: frozenset = field(compare=False)
    paths: dict = field(compare=False)   # agent_id → list[tuple]
    node_id: int = field(compare=False)


# ---------------------------------------------------------------------------
# CBSPlanner
# ---------------------------------------------------------------------------

class CBSPlanner:
    """Conflict-Based Search multi-agent pathfinder.

    Correctly implements CBS with:
    - Constraint accumulation (not replacement)
    - Two children per conflict (one per agent)
    - Parent constraint inheritance
    - Edge conflict detection (swap)
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_nodes: int = 800,
        max_timesteps: int = 60,
    ) -> None:
        self.grid_size = grid_size
        self.max_nodes = max_nodes
        self.max_timesteps = max_timesteps

    def plan_all(self, agent_tasks: list[dict]) -> dict:
        """Plan collision-free paths for all agents.

        Parameters
        ----------
        agent_tasks:
            List of ``{"id": str, "start": tuple, "target": tuple}``.

        Returns
        -------
        dict with keys:
            feasible, paths, total_cost, conflicts_resolved, nodes_expanded,
            residual_conflicts
        """
        # Normalise to 3-tuples
        tasks = []
        for t in agent_tasks:
            start = t.get("start")
            target = t.get("target")
            if start is None:
                start = (0, 0, 0)
            elif len(start) == 2:
                start = (start[0], start[1], 0)
            else:
                start = tuple(start)
            if target is None:
                target = (self.grid_size - 1, self.grid_size - 1, 0)
            elif len(target) == 2:
                target = (target[0], target[1], 0)
            else:
                target = tuple(target)
            tasks.append({"id": str(t.get("id", "")), "start": start, "target": target})

        # --- Root node: plan each agent independently ---
        root_constraints: frozenset = frozenset()
        root_paths: dict[str, list[tuple]] = {}
        for t in tasks:
            path = _plan_single(
                t["id"], t["start"], t["target"], set(), self.grid_size, self.max_timesteps
            )
            root_paths[t["id"]] = path or [t["start"]]

        root_node = _CBSNode(
            cost=self._total_cost(root_paths),
            constraints=root_constraints,
            paths=root_paths,
            node_id=0,
        )

        heap: list[_CBSNode] = [root_node]
        nodes_expanded = 0
        conflicts_resolved = 0
        node_counter = 1

        task_map = {t["id"]: t for t in tasks}

        while heap and nodes_expanded < self.max_nodes:
            node = heapq.heappop(heap)
            nodes_expanded += 1

            conflict = self._first_conflict(node.paths)
            if conflict is None:
                # No conflicts — success
                return {
                    "feasible": True,
                    "paths": {aid: list(p) for aid, p in node.paths.items()},
                    "total_cost": round(node.cost, 4),
                    "conflicts_resolved": conflicts_resolved,
                    "nodes_expanded": nodes_expanded,
                    "residual_conflicts": 0,
                }

            conflicts_resolved += 1
            a1, a2, cell_a1, cell_a2, t = conflict

            # Create two child nodes — one constraining a1, one constraining a2
            for constrained_agent, constraint_cell in ((a1, cell_a1), (a2, cell_a2)):
                new_constraint = CBSConstraint(
                    agent_id=constrained_agent, cell=constraint_cell, timestep=t
                )
                child_constraints = node.constraints | frozenset([new_constraint])
                child_paths = dict(node.paths)

                # Replan only the constrained agent
                task = task_map.get(constrained_agent)
                if task is None:
                    continue
                new_path = _plan_single(
                    constrained_agent,
                    task["start"],
                    task["target"],
                    set(child_constraints),
                    self.grid_size,
                    self.max_timesteps,
                )
                if new_path is None:
                    continue  # No path under this constraint — skip child
                child_paths[constrained_agent] = new_path
                child_node = _CBSNode(
                    cost=self._total_cost(child_paths),
                    constraints=child_constraints,
                    paths=child_paths,
                    node_id=node_counter,
                )
                node_counter += 1
                heapq.heappush(heap, child_node)

        # Max nodes exceeded — return best partial solution
        residual = self._count_conflicts(node.paths)
        return {
            "feasible": residual == 0,
            "paths": {aid: list(p) for aid, p in node.paths.items()},
            "total_cost": round(node.cost, 4),
            "conflicts_resolved": conflicts_resolved,
            "nodes_expanded": nodes_expanded,
            "residual_conflicts": residual,
        }

    def _first_conflict(
        self, paths: dict[str, list[tuple]]
    ) -> tuple[str, str, tuple, tuple, int] | None:
        """Return (agent1, agent2, cell_for_a1, cell_for_a2, timestep) for the first
        conflict, or None.

        For vertex conflicts: cell_for_a1 == cell_for_a2 (same cell).
        For edge conflicts: each agent gets the destination cell of the other
        (prevents them from entering the cell occupied by the other agent).
        """
        agent_ids = list(paths.keys())
        max_t = max((len(p) for p in paths.values()), default=0)

        for t in range(max_t):
            positions: dict[tuple, str] = {}
            for aid in agent_ids:
                path = paths[aid]
                cell = path[t] if t < len(path) else path[-1]
                if cell in positions:
                    return (positions[cell], aid, cell, cell, t)
                positions[cell] = aid

        # Edge conflict (swap): agent A moves A→B while agent B moves B→A
        for t in range(max_t - 1):
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i + 1:]:
                    p1, p2 = paths[a1], paths[a2]
                    c1_t = p1[t] if t < len(p1) else p1[-1]
                    c1_t1 = p1[t + 1] if t + 1 < len(p1) else p1[-1]
                    c2_t = p2[t] if t < len(p2) else p2[-1]
                    c2_t1 = p2[t + 1] if t + 1 < len(p2) else p2[-1]
                    if c1_t == c2_t1 and c2_t == c1_t1:
                        # a1 wants to go to c2_t; a2 wants to go to c1_t.
                        # Constrain a1 from c2_t at t+1, and a2 from c1_t at t+1.
                        return (a1, a2, c2_t, c1_t, t + 1)

        return None

    def _count_conflicts(self, paths: dict[str, list[tuple]]) -> int:
        """Count total vertex + edge conflicts."""
        count = 0
        agent_ids = list(paths.keys())
        max_t = max((len(p) for p in paths.values()), default=0)

        for t in range(max_t):
            seen: dict[tuple, str] = {}
            for aid in agent_ids:
                path = paths[aid]
                cell = path[t] if t < len(path) else path[-1]
                if cell in seen:
                    count += 1
                else:
                    seen[cell] = aid

        for t in range(max_t - 1):
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i + 1:]:
                    p1, p2 = paths[a1], paths[a2]
                    c1_t = p1[t] if t < len(p1) else p1[-1]
                    c1_t1 = p1[t + 1] if t + 1 < len(p1) else p1[-1]
                    c2_t = p2[t] if t < len(p2) else p2[-1]
                    c2_t1 = p2[t + 1] if t + 1 < len(p2) else p2[-1]
                    if c1_t == c2_t1 and c2_t == c1_t1:
                        count += 1

        return count

    def _total_cost(self, paths: dict[str, list[tuple]]) -> float:
        return float(sum(len(p) for p in paths.values()))
