"""CRNAPlanner — A* path planning in CRNA space with obstacle avoidance.

Uses DynamicGrid for live CRNA values so it sees current obstacles
(cats, blocked APIs).  Zero external dependencies.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

from .dynamic_grid import get_grid


@dataclass(order=True)
class PathNode:
    f_val: float = field(compare=True)
    coord: tuple = field(compare=False)
    g_cost: float = field(compare=False, default=0.0)
    h_cost: float = field(compare=False, default=0.0)
    risk_so_far: float = field(compare=False, default=0.0)
    parent: Optional["PathNode"] = field(compare=False, default=None)

    def f_cost(self) -> float:  # noqa: D102
        return self.g_cost + self.h_cost


class CRNAPlanner:
    """A* path planner operating in CRNA-valued 3-D grid space."""

    def plan(
        self,
        start: tuple,
        target: tuple,
        risk_budget: float = 0.7,
        max_steps: int = 500,
    ) -> dict:
        """Find optimal path from *start* to *target* in CRNA space.

        Uses A* with a CRNA-aware cost function.  Cells whose current R
        exceeds *risk_budget* are skipped entirely.

        Returns
        -------
        dict with keys:
            found, path, total_cost, total_risk, steps, blocked_reason
        """
        grid = get_grid()

        def heuristic(a: tuple, b: tuple) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

        def move_cost(coord: tuple) -> float:
            v = grid.get(*coord)
            return v.c + v.r * 0.3

        open_heap: list[PathNode] = []
        start_node = PathNode(
            f_val=heuristic(start, target),
            coord=start,
            g_cost=0.0,
            h_cost=heuristic(start, target),
        )
        heapq.heappush(open_heap, start_node)

        came_from: dict[tuple, PathNode] = {start: start_node}
        g_scores: dict[tuple, float] = {start: 0.0}
        risk_scores: dict[tuple, float] = {start: 0.0}
        steps = 0

        while open_heap and steps < max_steps:
            current = heapq.heappop(open_heap)
            steps += 1

            if current.coord == target:
                # Reconstruct path
                path: list[tuple] = []
                node: Optional[PathNode] = current
                while node is not None:
                    path.append(node.coord)
                    node = node.parent
                path.reverse()
                return {
                    "found": True,
                    "path": path,
                    "total_cost": current.g_cost,
                    "total_risk": current.risk_so_far,
                    "steps": steps,
                    "blocked_reason": None,
                }

            for nb in self.neighbours(current.coord):
                cell = grid.get(*nb)
                if cell.r > risk_budget:
                    continue  # obstacle or too risky

                step_cost = move_cost(nb)
                tentative_g = current.g_cost + step_cost
                tentative_risk = current.risk_so_far + cell.r

                if nb not in g_scores or tentative_g < g_scores[nb]:
                    g_scores[nb] = tentative_g
                    risk_scores[nb] = tentative_risk
                    h = heuristic(nb, target)
                    nb_node = PathNode(
                        f_val=tentative_g + h,
                        coord=nb,
                        g_cost=tentative_g,
                        h_cost=h,
                        risk_so_far=tentative_risk,
                        parent=current,
                    )
                    came_from[nb] = nb_node
                    heapq.heappush(open_heap, nb_node)

        return {
            "found": False,
            "path": [],
            "total_cost": 0.0,
            "total_risk": 0.0,
            "steps": steps,
            "blocked_reason": "no_path_found" if steps < max_steps else "max_steps_exceeded",
        }

    def neighbours(self, coord: tuple) -> list[tuple]:
        """Return the 6 face-adjacent neighbours of a 3-D cell.

        Negative coordinates are filtered out.
        """
        x, y, z = coord
        result = []
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx >= 0 and ny >= 0 and nz >= 0:
                result.append((nx, ny, nz))
        return result

    def replan_needed(self, path: list[tuple], threshold: float = 0.3) -> bool:
        """Return True if any cell on *path* has R above *threshold*."""
        grid = get_grid()
        for coord in path:
            cell = grid.get(*coord)
            if cell.r > threshold:
                return True
        return False
