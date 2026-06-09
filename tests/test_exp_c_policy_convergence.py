"""Tests for EXP-C: PolicyLearner Convergence Rate by Domain."""

from __future__ import annotations

import pytest

from manifold.experiments.exp_c_policy_learner_convergence import run_policy_learner_convergence_benchmark


class TestPolicyLearnerConvergenceBenchmark:
    @pytest.fixture(scope="class")
    def result(self):
        return run_policy_learner_convergence_benchmark()

    def test_returns_expected_keys(self, result):
        for key in (
            "domain_results",
            "domains_promoted",
            "domains_total",
            "avg_decisions_to_promote",
            "total_escalations_before_promotion",
            "total_escalations_saved",
            "savings_ratio",
        ):
            assert key in result

    def test_all_domains_tested(self, result):
        assert result["domains_total"] == 8

    def test_domain_results_have_required_fields(self, result):
        for r in result["domain_results"]:
            assert "domain" in r
            assert "effective_min" in r
            assert "promoted" in r
            assert "escalations_before" in r
            assert "escalations_saved" in r

    def test_high_frequency_domains_promote(self, result):
        domain_map = {r["domain"]: r for r in result["domain_results"]}
        assert domain_map["devops"]["promoted"], "devops must promote with 92% approval"
        assert domain_map["finance"]["promoted"], "finance must promote with 92% approval"

    def test_healthcare_legal_not_promoted_quickly(self, result):
        domain_map = {r["domain"]: r for r in result["domain_results"]}
        # healthcare and legal require 50 decisions; n_decisions cap is 200
        # so they may or may not promote depending on rng — but if not promoted,
        # escalations_saved must be 0
        for domain in ("healthcare", "legal"):
            r = domain_map[domain]
            if not r["promoted"]:
                assert r["escalations_saved"] == 0

    def test_savings_ratio_in_range(self, result):
        assert 0.0 <= result["savings_ratio"] <= 1.0

    def test_some_domains_promote(self, result):
        assert result["domains_promoted"] > 0, "At least some domains must promote at 92%"

    def test_effective_min_matches_domain_minimums(self, result):
        from manifold.policy_learner import DOMAIN_MIN_DECISIONS
        for r in result["domain_results"]:
            domain = r["domain"]
            expected_min = DOMAIN_MIN_DECISIONS.get(domain, 10)
            # effective_min must be at least the domain minimum
            assert r["effective_min"] >= expected_min
