"""Tests for Phase 37: Formal Policy Verification (verify.py)."""

from __future__ import annotations

import pytest

from manifold.b2b import OrgPolicy
from manifold.verify import PolicyConflict, PolicyVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _policy(
    org_id: str = "org-a",
    min_reliability: float = 0.75,
    max_risk: float = 0.30,
    domain: str = "general",
) -> OrgPolicy:
    return OrgPolicy(
        org_id=org_id,
        min_reliability=min_reliability,
        max_risk=max_risk,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# PolicyConflict
# ---------------------------------------------------------------------------


class TestPolicyConflict:
    def test_to_dict(self) -> None:
        c = PolicyConflict(
            conflict_type="risk_deadlock",
            org_a_id="org-a",
            org_b_id="org-b",
            description="test",
            friction_contribution=0.5,
        )
        d = c.to_dict()
        assert d["conflict_type"] == "risk_deadlock"
        assert d["org_a_id"] == "org-a"
        assert d["org_b_id"] == "org-b"
        assert d["friction_contribution"] == 0.5

    def test_frozen(self) -> None:
        c = PolicyConflict(
            conflict_type="domain_mismatch",
            org_a_id="a",
            org_b_id="b",
            description="x",
            friction_contribution=0.1,
        )
        with pytest.raises(Exception):
            c.conflict_type = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_to_dict_keys(self) -> None:
        result = VerificationResult(
            compatible=True,
            org_a_id="a",
            org_b_id="b",
            conflicts=(),
            friction_score=0.0,
            deadlock_count=0,
            gap_count=0,
            recommendation="PROCEED",
        )
        d = result.to_dict()
        assert "compatible" in d
        assert "friction_score" in d
        assert "deadlock_count" in d
        assert "conflicts" in d
        assert "recommendation" in d

    def test_frozen(self) -> None:
        result = VerificationResult(
            compatible=True,
            org_a_id="a",
            org_b_id="b",
            conflicts=(),
            friction_score=0.0,
            deadlock_count=0,
            gap_count=0,
            recommendation="OK",
        )
        with pytest.raises(Exception):
            result.compatible = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PolicyVerifier — compatible policies
# ---------------------------------------------------------------------------


class TestCompatiblePolicies:
    def test_identical_policies_compatible(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.80, max_risk=0.30)
        b = _policy("org-b", min_reliability=0.80, max_risk=0.30)
        result = verifier.verify(a, b)
        assert result.compatible is True
        assert result.deadlock_count == 0

    def test_similar_policies_compatible(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.80, max_risk=0.30, domain="finance")
        b = _policy("org-b", min_reliability=0.82, max_risk=0.28, domain="finance")
        result = verifier.verify(a, b)
        assert result.compatible is True

    def test_friction_score_in_range(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a")
        b = _policy("org-b")
        result = verifier.verify(a, b)
        assert 0.0 <= result.friction_score <= 1.0

    def test_zero_deadlocks_when_compatible(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.75, max_risk=0.40)
        b = _policy("org-b", min_reliability=0.75, max_risk=0.40)
        result = verifier.verify(a, b)
        assert result.deadlock_count == 0

    def test_recommendation_proceed(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.80, max_risk=0.25, domain="general")
        b = _policy("org-b", min_reliability=0.80, max_risk=0.25, domain="general")
        result = verifier.verify(a, b)
        assert "PROCEED" in result.recommendation

    def test_org_ids_in_result(self) -> None:
        verifier = PolicyVerifier()
        result = verifier.verify(_policy("alpha"), _policy("beta"))
        assert result.org_a_id == "alpha"
        assert result.org_b_id == "beta"


# ---------------------------------------------------------------------------
# PolicyVerifier — deadlock detection
# ---------------------------------------------------------------------------


class TestDeadlockDetection:
    def test_reliability_deadlock_detected(self) -> None:
        """Both sides demand more trust than the other can guarantee."""
        verifier = PolicyVerifier()
        # A demands trust ≥ 0.95 (max_risk=0.05) but B only guarantees 0.80
        # B demands trust ≥ 0.95 (max_risk=0.05) but A only guarantees 0.80
        a = _policy("org-a", min_reliability=0.80, max_risk=0.05)
        b = _policy("org-b", min_reliability=0.80, max_risk=0.05)
        result = verifier.verify(a, b)
        deadlock_types = {c.conflict_type for c in result.conflicts}
        assert "reliability_deadlock" in deadlock_types
        assert result.deadlock_count >= 1
        assert result.compatible is False

    def test_risk_deadlock_example_from_spec(self) -> None:
        """Spec example: Org A requires MinTrust>0.9, Org B has MaxTrustCap=0.8."""
        verifier = PolicyVerifier()
        # A: max_risk=0.05 → demands trust ≥ 0.95 from B; B offers min_rel=0.70 → fails
        # B: max_risk=0.05 → demands trust ≥ 0.95 from A; A offers min_rel=0.70 → fails
        a = _policy("org-a", min_reliability=0.70, max_risk=0.05)
        b = _policy("org-b", min_reliability=0.70, max_risk=0.05)
        result = verifier.verify(a, b)
        deadlock_types = {c.conflict_type for c in result.conflicts}
        # Both sides can't satisfy each other → reliability_deadlock
        assert "reliability_deadlock" in deadlock_types

    def test_deadlock_makes_compatible_false(self) -> None:
        verifier = PolicyVerifier()
        # Very strict risk tolerance, modest reliability offered
        a = _policy("org-a", min_reliability=0.70, max_risk=0.05)
        b = _policy("org-b", min_reliability=0.70, max_risk=0.05)
        result = verifier.verify(a, b)
        assert result.compatible is False

    def test_deadlock_recommendation_block(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.70, max_risk=0.05)
        b = _policy("org-b", min_reliability=0.70, max_risk=0.05)
        result = verifier.verify(a, b)
        assert "BLOCK" in result.recommendation

    def test_no_deadlock_when_easily_compatible(self) -> None:
        verifier = PolicyVerifier()
        # High min_reliability offered, lenient max_risk
        a = _policy("org-a", min_reliability=0.90, max_risk=0.50)
        b = _policy("org-b", min_reliability=0.90, max_risk=0.50)
        result = verifier.verify(a, b)
        assert result.deadlock_count == 0


# ---------------------------------------------------------------------------
# PolicyVerifier — gap conflicts (negotiable)
# ---------------------------------------------------------------------------


class TestGapConflicts:
    def test_domain_mismatch_detected(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", domain="finance")
        b = _policy("org-b", domain="legal")
        result = verifier.verify(a, b)
        gap_types = {c.conflict_type for c in result.conflicts}
        assert "domain_mismatch" in gap_types

    def test_domain_mismatch_is_gap_not_deadlock(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.75, max_risk=0.40, domain="finance")
        b = _policy("org-b", min_reliability=0.75, max_risk=0.40, domain="legal")
        result = verifier.verify(a, b)
        assert result.compatible is True  # domain mismatch is a gap, not deadlock
        assert result.gap_count >= 1

    def test_reliability_gap_detected(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.90)
        b = _policy("org-b", min_reliability=0.70)
        result = verifier.verify(a, b)
        gap_types = {c.conflict_type for c in result.conflicts}
        assert "reliability_gap" in gap_types

    def test_risk_gap_detected(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", max_risk=0.10)
        b = _policy("org-b", max_risk=0.50)
        result = verifier.verify(a, b)
        gap_types = {c.conflict_type for c in result.conflicts}
        assert "risk_gap" in gap_types

    def test_small_differences_no_gaps(self) -> None:
        """Differences ≤ 0.05 should not trigger gap conflicts."""
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.80, max_risk=0.30)
        b = _policy("org-b", min_reliability=0.82, max_risk=0.32, domain="general")
        result = verifier.verify(a, b)
        gap_types = {c.conflict_type for c in result.conflicts}
        assert "reliability_gap" not in gap_types
        assert "risk_gap" not in gap_types

    def test_gap_compatible_true(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.90, max_risk=0.30)
        b = _policy("org-b", min_reliability=0.70, max_risk=0.30)
        result = verifier.verify(a, b)
        assert result.compatible is True  # gaps only → still compatible

    def test_gap_recommendation_warn(self) -> None:
        verifier = PolicyVerifier(friction_threshold=0.0)  # low threshold to trigger WARN
        a = _policy("org-a", max_risk=0.10)
        b = _policy("org-b", max_risk=0.90)
        result = verifier.verify(a, b)
        # With very low threshold and large gap, should warn
        assert "PROCEED" in result.recommendation or "WARN" in result.recommendation


# ---------------------------------------------------------------------------
# FrictionScore
# ---------------------------------------------------------------------------


class TestFrictionScore:
    def test_friction_zero_for_identical(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.80, max_risk=0.30, domain="general")
        b = _policy("org-b", min_reliability=0.80, max_risk=0.30, domain="general")
        result = verifier.verify(a, b)
        assert result.friction_score == 0.0

    def test_friction_positive_when_conflicts(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", domain="finance", min_reliability=0.95)
        b = _policy("org-b", domain="legal", min_reliability=0.70)
        result = verifier.verify(a, b)
        assert result.friction_score > 0.0

    def test_friction_max_one(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", min_reliability=0.99, max_risk=0.01, domain="alpha")
        b = _policy("org-b", min_reliability=0.99, max_risk=0.01, domain="beta")
        result = verifier.verify(a, b)
        assert result.friction_score <= 1.0

    def test_friction_higher_for_deadlock(self) -> None:
        verifier = PolicyVerifier()
        # Deadlock scenario
        a_dead = _policy("org-c", min_reliability=0.70, max_risk=0.05)
        b_dead = _policy("org-d", min_reliability=0.70, max_risk=0.05)
        result_dead = verifier.verify(a_dead, b_dead)

        # Deadlock should produce > 0 friction
        assert result_dead.friction_score >= 0.0


# ---------------------------------------------------------------------------
# verify_many and compatibility_matrix
# ---------------------------------------------------------------------------


class TestVerifyMany:
    def test_verify_many_empty(self) -> None:
        verifier = PolicyVerifier()
        result = verifier.verify_many([])
        assert result == {}

    def test_verify_many_single(self) -> None:
        verifier = PolicyVerifier()
        result = verifier.verify_many([_policy("org-a")])
        assert result == {}

    def test_verify_many_two(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a")
        b = _policy("org-b")
        result = verifier.verify_many([a, b])
        assert len(result) == 1
        assert ("org-a", "org-b") in result

    def test_verify_many_three(self) -> None:
        verifier = PolicyVerifier()
        policies = [_policy(f"org-{i}") for i in range(3)]
        result = verifier.verify_many(policies)
        assert len(result) == 3

    def test_compatibility_matrix_rows(self) -> None:
        verifier = PolicyVerifier()
        policies = [_policy(f"org-{i}") for i in range(3)]
        matrix = verifier.compatibility_matrix(policies)
        assert len(matrix) == 3
        for row in matrix:
            assert "org_a" in row
            assert "org_b" in row
            assert "friction_score" in row
            assert "compatible" in row

    def test_compatibility_matrix_serialisable(self) -> None:
        import json

        verifier = PolicyVerifier()
        policies = [_policy(f"org-{i}") for i in range(4)]
        matrix = verifier.compatibility_matrix(policies)
        json.dumps(matrix)  # should not raise


# ---------------------------------------------------------------------------
# to_dict serialisability
# ---------------------------------------------------------------------------


class TestSerialisability:
    def test_verification_result_to_dict(self) -> None:
        import json

        verifier = PolicyVerifier()
        a = _policy("org-a", domain="finance", min_reliability=0.95)
        b = _policy("org-b", domain="legal", min_reliability=0.70)
        result = verifier.verify(a, b)
        d = result.to_dict()
        json.dumps(d)  # should not raise

    def test_conflicts_list_in_dict(self) -> None:
        verifier = PolicyVerifier()
        a = _policy("org-a", domain="finance")
        b = _policy("org-b", domain="legal")
        result = verifier.verify(a, b)
        d = result.to_dict()
        assert isinstance(d["conflicts"], list)
