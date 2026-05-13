"""Tests for kinodynamic planning for Roomba (PROMPT D3)."""
import math

import pytest

from manifold.dynamic_grid import DynamicGrid, get_grid
from manifold.kinodynamic_planner import KinodynamicPlanner, KinodynamicState


class TestKinodynamicPlanner:
    def setup_method(self):
        """Set up a clean grid for each test."""
        # Populate the grid with traversable cells
        grid = get_grid()
        for x in range(10):
            for y in range(10):
                grid.set_base(x, y, 0, c=0.2, r=0.1, n=0.5, a=0.5)

    def test_plan_returns_feasible_path(self):
        """plan_kinodynamic returns a feasible path (found=True)."""
        planner = KinodynamicPlanner()
        result = planner.plan_kinodynamic(start=(0, 0, 0), target=(5, 5, 0))
        assert result["found"] is True, f"Expected found=True, got: {result}"
        assert len(result["path"]) > 0
        assert result["kinodynamically_feasible"] is True

    def test_estimated_duration_formula(self):
        """estimated_duration_seconds = path_length * CELL_SIZE / MAX_VEL."""
        planner = KinodynamicPlanner()
        result = planner.plan_kinodynamic(start=(0, 0, 0), target=(4, 0, 0))
        assert result["found"] is True
        path_length = len(result["path"]) - 1
        expected_duration = path_length * planner.CELL_SIZE / planner.MAX_VEL
        assert abs(result["estimated_duration_seconds"] - expected_duration) < 0.01, (
            f"Expected duration {expected_duration:.3f}, got {result['estimated_duration_seconds']}"
        )

    def test_straight_line_path_minimises_heading_changes(self):
        """A straight-line path has fewer heading changes than a zigzag path."""
        planner = KinodynamicPlanner()
        # Straight line: (0,0,0) → (5,0,0) should have 0 heading changes
        straight = planner.plan_kinodynamic(start=(0, 0, 0), target=(5, 0, 0))
        assert straight["found"] is True
        # A diagonal target will force heading changes
        diagonal = planner.plan_kinodynamic(start=(0, 0, 0), target=(3, 3, 0))
        assert diagonal["found"] is True
        # Straight path should have fewer heading changes than diagonal
        assert straight["total_heading_changes"] <= diagonal["total_heading_changes"]

    def test_initial_heading_degrees_used(self):
        """plan with initial_theta correctly orients from the start."""
        planner = KinodynamicPlanner()
        # North-facing robot at (5, 0) heading to (5, 4)
        result_north = planner.plan_kinodynamic(
            start=(5, 0, 0), target=(5, 4, 0), initial_theta=math.pi / 2
        )
        # East-facing robot at (5, 0) heading to (5, 4) — must turn first
        result_east = planner.plan_kinodynamic(
            start=(5, 0, 0), target=(5, 4, 0), initial_theta=0.0
        )
        assert result_north["found"] is True
        assert result_east["found"] is True
        # Both should find a path; north-facing should have fewer heading changes
        assert result_north["total_heading_changes"] <= result_east["total_heading_changes"]
