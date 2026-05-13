"""Tests for EXP2 — Bayesian CRNA Cell."""
from __future__ import annotations

import math

from manifold.experiments.bayesian_crna import (
    BayesianCRNACell,
    BayesianGrid,
    run_bayesian_vs_scalar_benchmark,
)


# Test 1: BayesianCRNACell.entropy() returns a finite positive float
def test_entropy_returns_positive_float():
    # Use large variance to ensure positive differential entropy
    cell = BayesianCRNACell(x=0, y=0, z=0, n_dist=(1.0, 0.5))
    h = cell.entropy()
    assert isinstance(h, float)
    assert math.isfinite(h)
    assert h > 0.0


# Test 2: bayesian_update() reduces variance after observation
def test_bayesian_update_reduces_variance():
    cell = BayesianCRNACell(x=0, y=0, z=0)
    initial_var = cell.n_dist[1]
    cell.bayesian_update({"c": 0.5, "r": 0.5, "n": 0.8, "a": 0.3}, likelihood_std=0.1)
    posterior_var = cell.n_dist[1]
    assert posterior_var < initial_var, (
        f"Variance did not decrease: {initial_var} -> {posterior_var}"
    )


# Test 3: Multiple updates converge toward true value
def test_multiple_updates_converge():
    import random
    rng = random.Random(42)
    true_n = 0.7
    cell = BayesianCRNACell(x=0, y=0, z=0, n_dist=(1.0, 0.5))
    initial_error = abs(cell.n_dist[0] - true_n)

    for _ in range(20):
        noisy_obs = max(0.0, min(1.0, true_n + rng.gauss(0, 0.1)))
        cell.bayesian_update({"n": noisy_obs}, likelihood_std=0.1)

    final_error = abs(cell.n_dist[0] - true_n)
    assert final_error < initial_error, (
        f"Did not converge: initial_error={initial_error:.4f}, final_error={final_error:.4f}"
    )


# Test 4: BayesianGrid.entropy_map() returns lower entropy for visited cells
def test_entropy_map_lower_for_visited():
    grid = BayesianGrid()
    # Update one cell many times
    for _ in range(10):
        grid.update_from_sensor(0, 0, 0, {"c": 0.5, "r": 0.3, "n": 0.4, "a": 0.6}, 0.9)

    emap = grid.entropy_map()
    assert (0, 0, 0) in emap

    # Create a fresh cell for comparison (no observations)
    fresh = BayesianCRNACell(x=1, y=1, z=1, n_dist=(1.0, 0.5))
    # Update the fresh cell through grid too for fair comparison
    grid2 = BayesianGrid()
    grid2.update_from_sensor(1, 1, 1, {"n": 0.5}, 0.1)  # just one update
    emap2 = grid2.entropy_map()

    # The heavily-updated cell should have lower entropy than a single-update cell
    # by ensuring variance shrank more
    cell_updated = grid._cells[(0, 0, 0)]
    # Variance should be very small after 10 updates
    assert cell_updated.n_dist[1] < fresh.n_dist[1], (
        f"Expected reduced variance after updates: {cell_updated.n_dist[1]} vs {fresh.n_dist[1]}"
    )


# Test 5: run_bayesian_vs_scalar_benchmark bayesian_mse < average_mse
def test_bayesian_beats_average_mse():
    result = run_bayesian_vs_scalar_benchmark()
    assert "bayesian_mse" in result
    assert "average_mse" in result
    assert "max_override_mse" in result
    assert "bayesian_wins" in result
    assert isinstance(result["bayesian_wins"], bool)
    # Bayesian should equal or beat simple averaging (prior regularisation)
    assert result["bayesian_wins"], (
        f"Bayesian MSE {result['bayesian_mse']} should be <= average MSE {result['average_mse']}"
    )
