"""Tests for EXP-A: Bayesian CRNA + CBS Integration."""

from __future__ import annotations

import pytest

from manifold.experiments.exp_a_bayesian_cbs import run_bayesian_cbs_benchmark


class TestBayesianCBSBenchmark:
    @pytest.fixture(scope="class")
    def result(self):
        return run_bayesian_cbs_benchmark()

    def test_returns_expected_keys(self, result):
        assert "high_entropy_metrics" in result
        assert "low_entropy_metrics" in result
        assert "entropy_reduction" in result
        assert "risk_improvement" in result
        assert "cost_improvement" in result
        assert "hypothesis_confirmed" in result

    def test_feasibility_keys_present(self, result):
        assert "feasible" in result["high_entropy_metrics"]
        assert "feasible" in result["low_entropy_metrics"]

    def test_entropy_reduced_by_sensor_observations(self, result):
        assert result["entropy_reduction"] > 0, (
            "Low-entropy grid (200 sensor updates) must have lower entropy "
            "than high-entropy grid (0 updates)"
        )

    def test_metric_values_in_range(self, result):
        for label in ("high_entropy_metrics", "low_entropy_metrics"):
            m = result[label]
            assert 0.0 <= m["avg_cost"] <= 1.0
            assert 0.0 <= m["avg_risk"] <= 1.0
            assert isinstance(m["mean_grid_entropy"], float)  # differential entropy can be negative

    def test_hypothesis_confirmed_type(self, result):
        assert isinstance(result["hypothesis_confirmed"], bool)
