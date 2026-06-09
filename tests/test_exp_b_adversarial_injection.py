"""Tests for EXP-B: Adversarial Brain Injection Security Test."""

from __future__ import annotations

import pytest

from manifold.experiments.exp_b_adversarial_injection import run_adversarial_injection_benchmark


class TestAdversarialInjectionBenchmark:
    @pytest.fixture(scope="class")
    def result(self):
        return run_adversarial_injection_benchmark()

    def test_returns_expected_keys(self, result):
        for key in (
            "pattern_catch_rate",
            "paraphrase_catch_rate",
            "encoding_catch_rate",
            "benign_false_positive_rate",
            "overall_adversarial_catch_rate",
            "security_margin",
            "layer_breakdown",
            "adversarial_misses",
            "results",
        ):
            assert key in result

    def test_pattern_catch_rate_high(self, result):
        assert result["pattern_catch_rate"] >= 0.75, (
            "Direct pattern attacks should be caught at ≥75%"
        )

    def test_benign_false_positive_rate_low(self, result):
        assert result["benign_false_positive_rate"] <= 0.20, (
            "Benign messages must not be blocked at rate >20%"
        )

    def test_overall_catch_rate_positive(self, result):
        assert result["overall_adversarial_catch_rate"] > 0.0

    def test_security_margin_matches_overall(self, result):
        assert result["security_margin"] == result["overall_adversarial_catch_rate"]

    def test_results_list_has_entries(self, result):
        assert len(result["results"]) > 0

    def test_results_have_required_fields(self, result):
        for r in result["results"]:
            assert "message" in r
            assert "category" in r
            assert "intercepted" in r
            assert "action" in r
            assert "risk_score" in r

    def test_rates_are_fractions(self, result):
        for key in ("pattern_catch_rate", "paraphrase_catch_rate",
                    "encoding_catch_rate", "benign_false_positive_rate",
                    "overall_adversarial_catch_rate"):
            assert 0.0 <= result[key] <= 1.0, f"{key} must be in [0,1]"
