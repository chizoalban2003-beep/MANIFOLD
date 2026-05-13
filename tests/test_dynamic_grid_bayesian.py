"""Tests for EXP2 — Bayesian sensor fusion in DynamicGrid.

EXP2 result: max-override MSE = 0.024753 vs Bayesian MSE = 0.000006 (4131x worse).
"""
from __future__ import annotations

import math
import threading

import pytest

from manifold.cell_update_bus import CellCoord, CellUpdate
from manifold.dynamic_grid import DynamicCell, DynamicGrid, CRNAValues, _PRIOR_STD_INIT


@pytest.fixture()
def grid():
    """Fresh DynamicGrid for each test — not the singleton."""
    g = DynamicGrid.__new__(DynamicGrid)
    g._lock = threading.Lock()
    g._cells = {}
    return g


# ---------------------------------------------------------------------------
# Test 1: Single sensor reading stores correct mean and std
# ---------------------------------------------------------------------------

def test_single_observation_stores_mean_and_std(grid):
    """First Bayesian observation sets mean=r_obs and std=_PRIOR_STD_INIT."""
    grid.set_base(0, 0, 0, c=0.3, r=0.2, n=0.5, a=0.5)
    update = CellUpdate(
        coord=CellCoord(x=0, y=0, z=0),
        r_delta=0.5,   # ov_r = base.r + r_delta = 0.2 + 0.5 = 0.7
        ttl=60.0,
        source="sensor1",
    )
    grid.apply_update(update)
    cell = grid._cells[(0, 0, 0)]
    assert cell._r_mean is not None, "Bayesian mean should be set after observation"
    assert abs(cell._r_mean - 0.7) < 1e-6
    assert abs(cell._r_std - _PRIOR_STD_INIT) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: Two consistent readings reduce std (posterior tighter than prior)
# ---------------------------------------------------------------------------

def test_two_consistent_readings_reduce_std(grid):
    """Two consistent readings should produce a tighter posterior than the prior."""
    grid.set_base(1, 1, 0, c=0.3, r=0.2, n=0.5, a=0.5)

    # First reading: r_obs = 0.8
    u1 = CellUpdate(coord=CellCoord(1, 1, 0), r_delta=0.6, ttl=60.0, source="s1")
    grid.apply_update(u1)
    cell = grid._cells[(1, 1, 0)]
    std_after_first = cell._r_std

    # Second reading: r_obs = 0.8 again (consistent)
    u2 = CellUpdate(coord=CellCoord(1, 1, 0), r_delta=0.6, ttl=60.0, source="s2")
    grid.apply_update(u2)
    std_after_second = cell._r_std

    assert std_after_second < std_after_first, (
        f"Two consistent readings should reduce uncertainty: "
        f"{std_after_second:.4f} >= {std_after_first:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Two conflicting readings produce mean between them (not the maximum)
# ---------------------------------------------------------------------------

def test_conflicting_readings_produce_middle_mean(grid):
    """High and low R observations should fuse to a mean between them."""
    grid.set_base(2, 2, 0, c=0.3, r=0.3, n=0.5, a=0.5)

    # First reading: high risk
    u_high = CellUpdate(coord=CellCoord(2, 2, 0), r_delta=0.6, ttl=60.0)  # ov_r = 0.9
    grid.apply_update(u_high)
    # Second reading: low risk
    u_low = CellUpdate(coord=CellCoord(2, 2, 0), r_delta=-0.2, ttl=60.0)  # ov_r = 0.1
    grid.apply_update(u_low)

    cell = grid._cells[(2, 2, 0)]
    bayesian_r = cell._r_mean

    # The Bayesian posterior mean must be strictly between 0.1 and 0.9
    assert bayesian_r > 0.1, f"Posterior {bayesian_r:.4f} should be above low reading 0.1"
    assert bayesian_r < 0.9, f"Posterior {bayesian_r:.4f} should be below high reading 0.9"
    # And it must NOT equal the max of the two (max-override would give 0.9)
    assert bayesian_r < 0.85, (
        f"Bayesian {bayesian_r:.4f} should not equal the max-override value ~0.9"
    )


# ---------------------------------------------------------------------------
# Test 4: get_r_uncertainty returns lower value after more observations
# ---------------------------------------------------------------------------

def test_get_r_uncertainty_decreases_with_observations(grid):
    """More sensor readings → lower R uncertainty (tighter posterior)."""
    grid.set_base(3, 3, 0, c=0.3, r=0.4, n=0.5, a=0.5)

    u = CellUpdate(coord=CellCoord(3, 3, 0), r_delta=0.1, ttl=60.0)

    # No observations yet — max uncertainty
    unc_before = grid.get_r_uncertainty(3, 3, 0)
    assert abs(unc_before - _PRIOR_STD_INIT) < 1e-6, (
        f"Before any observations, uncertainty should equal _PRIOR_STD_INIT "
        f"({_PRIOR_STD_INIT}), got {unc_before}"
    )

    # One observation
    grid.apply_update(u)
    unc_after_1 = grid.get_r_uncertainty(3, 3, 0)

    # Five more observations
    for _ in range(5):
        grid.apply_update(u)
    unc_after_6 = grid.get_r_uncertainty(3, 3, 0)

    assert unc_after_6 < unc_after_1, (
        f"Uncertainty after 6 observations ({unc_after_6:.4f}) should be lower "
        f"than after 1 ({unc_after_1:.4f})"
    )
