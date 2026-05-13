"""manifold/kinodynamic_planner.py — Physics-aware path planning for Roomba.

Extends CRNAPlanner with kinodynamic constraints for an iRobot Roomba-class
robot.  Paths must be physically achievable within the robot's motion envelope.

Roomba physical constraints (iRobot specs):
  max_velocity:       0.5 m/s
  min_turning_radius: 0.18 m (can turn on spot → effectively 0 for planning)
  max_angular_velocity: 2.0 rad/s
  cell_size:          0.5 m (each CRNA cell is 0.5 × 0.5 metres)

Since the Roomba can turn on the spot (differential drive), all adjacent cells
are kinodynamically reachable.  The planner adds a small angular cost for
heading changes to bias the path towards straight-line motion.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .planner import CRNAPlanner
from .dynamic_grid import get_grid


@dataclass
class KinodynamicState:
    """6-DOF kinematic state of the Roomba (projected to 3-D voxel grid)."""

    x: float
    y: float
    z: float
    theta: float = 0.0      # heading in radians (XY-plane)
    velocity: float = 0.0   # current speed (m/s)


class KinodynamicPlanner(CRNAPlanner):
    """A* path planner with kinodynamic costs for a Roomba-class robot.

    Extends :class:`CRNAPlanner` by adding a heading-change penalty so the
    planner naturally prefers straight-line segments over unnecessary turns.

    Class constants
    ---------------
    CELL_SIZE:
        Physical size of one CRNA grid cell in metres.  Default 0.5 m.
    MAX_VEL:
        Maximum Roomba velocity in m/s.  Default 0.5 m/s.
    MAX_OMEGA:
        Maximum angular velocity in rad/s.  Default 2.0 rad/s.
    HEADING_CHANGE_COST:
        Cost multiplier applied per radian of heading change.  Default 0.1.
    """

    CELL_SIZE: float = 0.5   # metres per cell
    MAX_VEL: float = 0.5     # m/s
    MAX_OMEGA: float = 2.0   # rad/s
    HEADING_CHANGE_COST: float = 0.1  # cost per radian of heading change

    # Direction → heading angle (radians) in the XY plane
    _DIR_HEADING: dict[tuple[int, int, int], float] = {
        (1, 0, 0): 0.0,
        (-1, 0, 0): math.pi,
        (0, 1, 0): math.pi / 2,
        (0, -1, 0): -math.pi / 2,
        (0, 0, 1): 0.0,   # vertical — no heading change cost
        (0, 0, -1): 0.0,
    }

    def _heading_delta(
        self, prev_dir: tuple[int, int, int], new_dir: tuple[int, int, int]
    ) -> float:
        """Return the absolute heading change (radians) between two grid directions."""
        prev_theta = self._DIR_HEADING.get(prev_dir, 0.0)
        new_theta = self._DIR_HEADING.get(new_dir, 0.0)
        delta = abs(new_theta - prev_theta)
        # Normalise to [0, π]
        if delta > math.pi:
            delta = 2 * math.pi - delta
        return delta

    def feasible_transitions(
        self,
        state: KinodynamicState,
        target_cell: tuple,
    ) -> list[KinodynamicState]:
        """Return list of kinodynamically reachable next states.

        For the Roomba (differential drive, can turn on spot), all adjacent
        cells are reachable.  The heading is updated to point toward the target.

        Parameters
        ----------
        state:
            Current kinematic state.
        target_cell:
            (x, y, z) of the desired next cell.
        """
        dx = target_cell[0] - int(state.x)
        dy = target_cell[1] - int(state.y)
        # Heading change in XY plane (z-component not used for heading)
        new_theta = math.atan2(dy, dx) if (dx != 0 or dy != 0) else state.theta
        new_state = KinodynamicState(
            x=float(target_cell[0]),
            y=float(target_cell[1]),
            z=float(target_cell[2]),
            theta=new_theta,
            velocity=self.MAX_VEL,
        )
        return [new_state]

    def plan_kinodynamic(
        self,
        start: tuple,
        target: tuple,
        initial_theta: float = 0.0,
        risk_budget: float = 0.7,
    ) -> dict:
        """A* path planning with kinodynamic heading-change costs.

        Parameters
        ----------
        start:
            Starting (x, y, z) cell.
        target:
            Target (x, y, z) cell.
        initial_theta:
            Initial heading in radians.  Default 0.0 (east).
        risk_budget:
            Maximum risk per cell.  Default 0.7.

        Returns
        -------
        dict with all keys from :meth:`CRNAPlanner.plan` plus:
            ``total_heading_changes`` (int),
            ``estimated_duration_seconds`` (float),
            ``kinodynamically_feasible`` (bool)
        """
        grid = get_grid()
        start = tuple(int(v) for v in start)
        target = tuple(int(v) for v in target)

        def heuristic(a: tuple, b: tuple) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

        # State: (f, g, risk, heading_changes, coord, prev_dir)
        open_heap: list = []
        start_entry = (
            heuristic(start, target), 0.0, 0.0, 0, start, (0, 0, 0)
        )
        heapq.heappush(open_heap, start_entry)
        g_scores: dict = {start: 0.0}
        came_from: dict = {}   # coord -> (parent_coord, prev_dir, heading_changes)
        heading_changes_map: dict = {start: 0}

        steps = 0
        while open_heap and steps < 5000:
            f, g, risk, hc, pos, prev_dir = heapq.heappop(open_heap)
            steps += 1

            if pos == target:
                # Reconstruct path
                path: list[tuple] = []
                cur = pos
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur][0]
                path.append(start)
                path.reverse()
                total_heading_changes = heading_changes_map.get(target, 0)
                path_length = len(path) - 1
                est_duration = (path_length * self.CELL_SIZE) / self.MAX_VEL
                return {
                    "found": True,
                    "path": path,
                    "total_cost": round(g, 4),
                    "total_risk": round(risk, 4),
                    "steps": steps,
                    "blocked_reason": "",
                    "total_heading_changes": total_heading_changes,
                    "estimated_duration_seconds": round(est_duration, 3),
                    "kinodynamically_feasible": True,
                }

            if g > g_scores.get(pos, float("inf")):
                continue

            x, y, z = pos
            for dx, dy, dz in [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ]:
                nb = (x + dx, y + dy, z + dz)
                if any(v < 0 for v in nb):
                    continue
                vals = grid.get(*nb)
                if vals.r > risk_budget:
                    continue
                new_dir = (dx, dy, dz)
                delta_theta = (
                    self._heading_delta(prev_dir, new_dir)
                    if prev_dir != (0, 0, 0) else 0.0
                )
                heading_change = 1 if delta_theta > 0.01 else 0
                move_cost = vals.c + vals.r * 0.3
                angular_cost = delta_theta * self.HEADING_CHANGE_COST
                new_g = g + move_cost + angular_cost
                new_risk = risk + vals.r
                new_hc = hc + heading_change
                if new_g < g_scores.get(nb, float("inf")):
                    g_scores[nb] = new_g
                    came_from[nb] = (pos, new_dir, new_hc)
                    heading_changes_map[nb] = new_hc
                    f_val = new_g + heuristic(nb, target)
                    heapq.heappush(open_heap, (f_val, new_g, new_risk, new_hc, nb, new_dir))

        return {
            "found": False,
            "path": [],
            "total_cost": 0.0,
            "total_risk": 0.0,
            "steps": steps,
            "blocked_reason": "no_path_found",
            "total_heading_changes": 0,
            "estimated_duration_seconds": 0.0,
            "kinodynamically_feasible": False,
        }

    def plan(
        self,
        start: tuple,
        target: tuple,
        risk_budget: float = 0.7,
        max_steps: int = 500,
    ) -> dict:
        """Drop-in override: delegates to plan_kinodynamic() with heading cost."""
        result = self.plan_kinodynamic(start, target, risk_budget=risk_budget)
        return result
