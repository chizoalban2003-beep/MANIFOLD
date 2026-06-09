"""Tests for EXP-D: Calibrated vs. Fixed Threshold Shadow A/B Test."""

from __future__ import annotations

import pytest

from manifold.experiments.exp_d_threshold_ab_test import run_threshold_ab_benchmark


class TestThresholdABBenchmark:
    @pytest.fixture(scope="class")
    def result(self):
        return run_threshold_ab_benchmark()

    def test_returns_expected_keys(self, result):
        for key in (
            "domain_results",
            "total_decisions",
            "total_divergences",
            "overall_divergence_rate",
            "avg_precision_gain",
            "avg_escalation_reduction",
            "calibration_worthwhile",
        ):
            assert key in result

    def test_total_decisions(self, result):
        assert result["total_decisions"] == 5000  # 5 domains × 1000

    def test_domain_results_coverage(self, result):
        domains = set(result["domain_results"].keys())
        assert domains == {"finance", "healthcare", "legal", "devops", "general"}

    def test_domain_result_fields(self, result):
        for domain, dr in result["domain_results"].items():
            assert "calibrated_threshold" in dr
            assert "fixed_threshold" in dr
            assert "fixed" in dr
            assert "calibrated" in dr
            assert "divergence_count" in dr
            assert "precision_gain" in dr
            assert "escalation_reduction" in dr

    def test_fixed_threshold_is_07(self, result):
        for dr in result["domain_results"].values():
            assert dr["fixed_threshold"] == 0.7

    def test_metrics_in_range(self, result):
        for dr in result["domain_results"].values():
            for variant in ("fixed", "calibrated"):
                m = dr[variant]
                assert 0.0 <= m["precision"] <= 1.0
                assert 0.0 <= m["recall"] <= 1.0
                assert 0.0 <= m["f1"] <= 1.0
                assert 0.0 <= m["escalation_rate"] <= 1.0

    def test_divergence_rate_in_range(self, result):
        assert 0.0 <= result["overall_divergence_rate"] <= 1.0

    def test_calibration_worthwhile_type(self, result):
        assert isinstance(result["calibration_worthwhile"], bool)
