"""Tests for Phase 61: Zero-Knowledge Policy Proofs (manifold/zkp.py).

Tests cover:
- Basic Schnorr proof generation and verification
- Fiat-Shamir challenge determinism
- Invalid / spoofed proof rejection
- Policy commitment generation and verification
- Edge cases: x=1, x=q-1, spoofed y, corrupted R, wrong context
"""

from __future__ import annotations

import pytest

from manifold.zkp import (
    PolicyCommitment,
    ZKPParams,
    ZKPVerifier,
    ZKProof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verifier() -> ZKPVerifier:
    return ZKPVerifier()


# ---------------------------------------------------------------------------
# ZKPParams
# ---------------------------------------------------------------------------


class TestZKPParams:
    def test_default_params(self) -> None:
        params = ZKPParams.default()
        assert params.g == 2
        assert params.p > 0
        assert params.q > 0
        # q = (p - 1) / 2
        assert params.p == 2 * params.q + 1

    def test_params_frozen(self) -> None:
        params = ZKPParams.default()
        with pytest.raises((AttributeError, TypeError)):
            params.g = 3  # type: ignore[misc]

    def test_p_is_large(self) -> None:
        params = ZKPParams.default()
        assert params.p.bit_length() >= 512

    def test_q_is_half_p_minus_one(self) -> None:
        params = ZKPParams.default()
        assert params.q == (params.p - 1) // 2


# ---------------------------------------------------------------------------
# ZKProof
# ---------------------------------------------------------------------------


class TestZKProof:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        v = _verifier()
        proof = v.prove(x=42, context="test")
        d = proof.to_dict()
        restored = ZKProof.from_dict(d)
        assert restored.y == proof.y
        assert restored.R == proof.R
        assert restored.s == proof.s
        assert restored.e == proof.e
        assert restored.context == proof.context

    def test_to_dict_uses_hex(self) -> None:
        v = _verifier()
        proof = v.prove(x=1, context="hex-test")
        d = proof.to_dict()
        assert d["y"].startswith("0x")
        assert d["R"].startswith("0x")

    def test_frozen(self) -> None:
        v = _verifier()
        proof = v.prove(x=7, context="frozen")
        with pytest.raises((AttributeError, TypeError)):
            proof.y = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ZKPVerifier — core prove / verify
# ---------------------------------------------------------------------------


class TestZKPVerifierProveVerify:
    def test_valid_proof_verifies(self) -> None:
        v = _verifier()
        proof = v.prove(x=100, context="demo")
        assert v.verify(proof)

    def test_x_equals_one(self) -> None:
        v = _verifier()
        proof = v.prove(x=1, context="edge-min")
        assert v.verify(proof)

    def test_large_x(self) -> None:
        v = _verifier()
        # Use a large x close to q-1 (the valid upper bound for the order-q subgroup)
        x = v.params.q - 2
        proof = v.prove(x=x, context="edge-max")
        assert v.verify(proof)

    def test_x_zero_raises(self) -> None:
        v = _verifier()
        with pytest.raises(ValueError, match="Secret x must be in"):
            v.prove(x=0, context="zero")

    def test_x_negative_raises(self) -> None:
        v = _verifier()
        with pytest.raises(ValueError, match="Secret x must be in"):
            v.prove(x=-1, context="negative")

    def test_x_equal_to_q_raises(self) -> None:
        v = _verifier()
        with pytest.raises(ValueError):
            v.prove(x=v.params.q, context="at-q")

    def test_two_proofs_same_x_differ_due_to_nonce(self) -> None:
        v = _verifier()
        p1 = v.prove(x=50, context="nonce")
        p2 = v.prove(x=50, context="nonce")
        # With overwhelming probability the random nonce r differs
        # so R and s should differ
        assert p1.R != p2.R or p1.s != p2.s

    def test_context_is_bound(self) -> None:
        v = _verifier()
        x = 77
        proof_a = v.prove(x=x, context="context-A")
        # Verify against context-A (should pass)
        assert v.verify(proof_a)

    def test_multiple_different_secrets(self) -> None:
        v = _verifier()
        for x in [1, 2, 100, 9999, 123456]:
            proof = v.prove(x=x, context=f"multi-{x}")
            assert v.verify(proof), f"Failed for x={x}"


# ---------------------------------------------------------------------------
# ZKPVerifier — anti-spoofing / false-positive tests
# ---------------------------------------------------------------------------


class TestZKPVerifierAntiSpoofing:
    """Ensure the verifier rejects tampered, spoofed, or random proofs."""

    def test_wrong_y_rejected(self) -> None:
        v = _verifier()
        proof = v.prove(x=42, context="spoof-y")
        bad = ZKProof(y=proof.y + 1, R=proof.R, s=proof.s, e=proof.e, context=proof.context)
        assert not v.verify(bad)

    def test_wrong_R_rejected(self) -> None:
        v = _verifier()
        proof = v.prove(x=42, context="spoof-R")
        bad = ZKProof(y=proof.y, R=proof.R + 1, s=proof.s, e=proof.e, context=proof.context)
        assert not v.verify(bad)

    def test_wrong_s_rejected(self) -> None:
        v = _verifier()
        proof = v.prove(x=42, context="spoof-s")
        bad = ZKProof(y=proof.y, R=proof.R, s=(proof.s + 1) % v.params.q, e=proof.e, context=proof.context)
        assert not v.verify(bad)

    def test_wrong_e_rejected(self) -> None:
        v = _verifier()
        proof = v.prove(x=42, context="spoof-e")
        bad = ZKProof(y=proof.y, R=proof.R, s=proof.s, e=(proof.e + 1) % v.params.q, context=proof.context)
        assert not v.verify(bad)

    def test_wrong_context_rejected(self) -> None:
        v = _verifier()
        proof = v.prove(x=42, context="real-context")
        bad = ZKProof(y=proof.y, R=proof.R, s=proof.s, e=proof.e, context="evil-context")
        assert not v.verify(bad)

    def test_all_zeros_rejected(self) -> None:
        v = _verifier()
        bad = ZKProof(y=0, R=0, s=0, e=0, context="zeros")
        assert not v.verify(bad)

    def test_y_equals_one_rejected(self) -> None:
        # y=1 would mean x=0 (identity element) — invalid
        v = _verifier()
        bad = ZKProof(y=1, R=2, s=1, e=1, context="y-one")
        assert not v.verify(bad)

    def test_random_values_rejected(self) -> None:
        """Random integers should not satisfy the proof equation."""
        import secrets as sec
        v = _verifier()
        p, q = v.params.p, v.params.q
        for _ in range(10):
            y = sec.randbelow(p - 2) + 2
            R = sec.randbelow(p - 2) + 2
            s = sec.randbelow(q)
            # Compute real e to test whether just plugging in random y/R/s works
            e = v._challenge(y, R, "random")  # noqa: SLF001
            # Tamper with s: changing s by 1 must break verification
            tampered = ZKProof(y=y, R=R, s=(s + 1) % q, e=e, context="random")
            assert not v.verify(tampered)

    def test_proof_for_different_x_does_not_verify_for_wrong_y(self) -> None:
        v = _verifier()
        proof_x10 = v.prove(x=10, context="ctx")
        proof_x20 = v.prove(x=20, context="ctx")
        # Swap y values: should fail because y no longer matches s
        bad = ZKProof(
            y=proof_x20.y,
            R=proof_x10.R,
            s=proof_x10.s,
            e=proof_x10.e,
            context=proof_x10.context,
        )
        assert not v.verify(bad)

    def test_recomputed_e_mismatch_rejects(self) -> None:
        v = _verifier()
        proof = v.prove(x=99, context="e-check")
        # Provide a wrong e (bypassed hash)
        wrong_e = (proof.e + 42) % v.params.q
        bad = ZKProof(y=proof.y, R=proof.R, s=proof.s, e=wrong_e, context=proof.context)
        assert not v.verify(bad)


# ---------------------------------------------------------------------------
# PolicyCommitment
# ---------------------------------------------------------------------------


class TestPolicyCommitment:
    def test_valid_commitment_verifies(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.2, risk=0.1, threshold=0.5)
        assert v.verify_policy_commitment(commitment)

    def test_commitment_to_dict_roundtrip(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.1, risk=0.05, threshold=0.4)
        d = commitment.to_dict()
        restored = PolicyCommitment.from_dict(d)
        assert v.verify_policy_commitment(restored)

    def test_zero_slack_raises(self) -> None:
        v = _verifier()
        with pytest.raises(ValueError, match="cost.*risk.*>=.*threshold"):
            v.commit_policy(cost=0.3, risk=0.3, threshold=0.5)

    def test_negative_slack_raises(self) -> None:
        v = _verifier()
        with pytest.raises(ValueError, match="cost.*risk.*>=.*threshold"):
            v.commit_policy(cost=0.4, risk=0.3, threshold=0.5)

    def test_commitment_threshold_in_context(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.1, risk=0.1, threshold=0.5)
        assert "0.500000" in commitment.proof.context

    def test_wrong_threshold_in_commitment_rejected(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.1, risk=0.1, threshold=0.5)
        # Tamper with threshold
        bad = PolicyCommitment(
            threshold=0.9,  # changed
            y=commitment.y,
            proof=commitment.proof,
            scale=commitment.scale,
        )
        assert not v.verify_policy_commitment(bad)

    def test_y_equals_one_rejected_by_policy_verifier(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.1, risk=0.1, threshold=0.5)
        bad_proof = ZKProof(
            y=1,
            R=commitment.proof.R,
            s=commitment.proof.s,
            e=commitment.proof.e,
            context=commitment.proof.context,
        )
        bad = PolicyCommitment(
            threshold=commitment.threshold,
            y=1,
            proof=bad_proof,
            scale=commitment.scale,
        )
        assert not v.verify_policy_commitment(bad)

    def test_mismatched_y_in_commitment_rejected(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.1, risk=0.1, threshold=0.5)
        bad = PolicyCommitment(
            threshold=commitment.threshold,
            y=commitment.y + 1,  # tampered
            proof=commitment.proof,
            scale=commitment.scale,
        )
        assert not v.verify_policy_commitment(bad)

    def test_different_costs_produce_different_commitments(self) -> None:
        v = _verifier()
        c1 = v.commit_policy(cost=0.1, risk=0.1, threshold=0.5)
        c2 = v.commit_policy(cost=0.2, risk=0.1, threshold=0.5)
        # Different x values → almost certainly different proofs
        assert c1.y != c2.y or c1.proof.R != c2.proof.R

    def test_full_cycle_low_risk_and_cost(self) -> None:
        v = _verifier()
        commitment = v.commit_policy(cost=0.05, risk=0.05, threshold=0.3)
        assert commitment.threshold == 0.3
        assert v.verify_policy_commitment(commitment)

    def test_commitment_frozen(self) -> None:
        v = _verifier()
        c = v.commit_policy(cost=0.1, risk=0.1, threshold=0.5)
        with pytest.raises((AttributeError, TypeError)):
            c.threshold = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


class TestZKPBatch:
    def test_prove_batch_all_verify(self) -> None:
        v = _verifier()
        pairs = [(1, "a"), (100, "b"), (999, "c")]
        proofs = v.prove_batch(pairs)
        assert all(v.verify(p) for p in proofs)

    def test_verify_batch_returns_per_proof(self) -> None:
        v = _verifier()
        valid = v.prove(x=42, context="v")
        bad = ZKProof(y=0, R=0, s=0, e=0, context="b")
        results = v.verify_batch([valid, bad])
        assert results[0] is True
        assert results[1] is False

    def test_empty_batch(self) -> None:
        v = _verifier()
        assert v.prove_batch([]) == []
        assert v.verify_batch([]) == []


# ---------------------------------------------------------------------------
# ZKPVerifier — summary
# ---------------------------------------------------------------------------


class TestZKPVerifierSummary:
    def test_summary_keys(self) -> None:
        v = _verifier()
        s = v.summary()
        assert "group_bits" in s
        assert "g" in s
        assert s["g"] == 2
        assert s["group_bits"] >= 512
