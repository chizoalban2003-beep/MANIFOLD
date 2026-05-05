"""Tests for Phase 57: Hardware Fingerprinting & Attestation (hardware.py)."""

from __future__ import annotations

import time

import pytest

from manifold.hardware import (
    ComputeChallenge,
    ComputeResponse,
    HardwareAttestor,
    HardwareProfile,
    ProofOfCompute,
)


# ---------------------------------------------------------------------------
# HardwareAttestor
# ---------------------------------------------------------------------------


class TestHardwareAttestor:
    def test_profile_returns_hardware_profile(self) -> None:
        attestor = HardwareAttestor()
        profile = attestor.profile()
        assert isinstance(profile, HardwareProfile)

    def test_fingerprint_is_hex_string(self) -> None:
        attestor = HardwareAttestor()
        fp = attestor.profile().fingerprint
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in fp)

    def test_profile_cached(self) -> None:
        attestor = HardwareAttestor()
        p1 = attestor.profile()
        p2 = attestor.profile()
        assert p1 is p2  # same object

    def test_fingerprint_stable(self) -> None:
        a1 = HardwareAttestor()
        a2 = HardwareAttestor()
        assert a1.profile().fingerprint == a2.profile().fingerprint

    def test_matches_self(self) -> None:
        attestor = HardwareAttestor()
        assert attestor.matches(attestor.profile().fingerprint)

    def test_does_not_match_garbage(self) -> None:
        attestor = HardwareAttestor()
        assert not attestor.matches("0" * 64)

    def test_profile_fields_populated(self) -> None:
        profile = HardwareAttestor().profile()
        assert profile.os_name
        assert profile.platform_system
        assert profile.python_version
        assert profile.python_implementation
        assert profile.cpu_count >= 0

    def test_profile_to_dict(self) -> None:
        profile = HardwareAttestor().profile()
        d = profile.to_dict()
        assert "fingerprint" in d
        assert "os_name" in d
        assert "cpu_count" in d

    def test_cpu_count_non_negative(self) -> None:
        assert HardwareAttestor().profile().cpu_count >= 0

    def test_python_version_looks_reasonable(self) -> None:
        pv = HardwareAttestor().profile().python_version
        parts = pv.split(".")
        assert len(parts) >= 2
        assert parts[0].isdigit()


# ---------------------------------------------------------------------------
# ComputeChallenge (data model)
# ---------------------------------------------------------------------------


class TestComputeChallenge:
    def _make(self, difficulty: int = 1, expired: bool = False) -> ComputeChallenge:
        now = time.time()
        return ComputeChallenge(
            challenge_id="cid-001",
            challenge_nonce="deadbeef",
            difficulty=difficulty,
            challenger_fingerprint="a" * 64,
            issued_at=now,
            expires_at=now + (-1 if expired else 300),
        )

    def test_not_expired(self) -> None:
        assert not self._make().is_expired

    def test_is_expired(self) -> None:
        assert self._make(expired=True).is_expired

    def test_to_dict_fields(self) -> None:
        c = self._make()
        d = c.to_dict()
        assert d["challenge_id"] == "cid-001"
        assert d["difficulty"] == 1
        assert "challenger_fingerprint" in d


# ---------------------------------------------------------------------------
# ProofOfCompute — issue / solve / verify
# ---------------------------------------------------------------------------


class TestProofOfCompute:
    def _poc(self, difficulty: int = 1) -> ProofOfCompute:
        return ProofOfCompute(
            HardwareAttestor(),
            default_difficulty=difficulty,
            challenge_ttl_seconds=300.0,
        )

    def test_issue_returns_challenge(self) -> None:
        poc = self._poc()
        challenge = poc.issue_challenge()
        assert isinstance(challenge, ComputeChallenge)

    def test_challenge_not_expired(self) -> None:
        poc = self._poc()
        challenge = poc.issue_challenge()
        assert not challenge.is_expired

    def test_challenge_id_non_empty(self) -> None:
        poc = self._poc()
        challenge = poc.issue_challenge()
        assert challenge.challenge_id

    def test_challenge_uses_default_difficulty(self) -> None:
        poc = self._poc(difficulty=2)
        challenge = poc.issue_challenge()
        assert challenge.difficulty == 2

    def test_challenge_override_difficulty(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=3)
        assert challenge.difficulty == 3

    def test_solve_difficulty_1(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        assert response.solution_hash.startswith("0")

    def test_verify_valid_response(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        assert poc.verify_response(challenge, response)

    def test_verify_wrong_nonce_fails(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        # Tamper with solution nonce
        tampered = ComputeResponse(
            challenge_id=response.challenge_id,
            prover_fingerprint=response.prover_fingerprint,
            solution_nonce=response.solution_nonce + 999,
            solution_hash=response.solution_hash,
            solved_at=response.solved_at,
            elapsed_seconds=response.elapsed_seconds,
        )
        assert not poc.verify_response(challenge, tampered)

    def test_verify_wrong_challenge_id_fails(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        tampered = ComputeResponse(
            challenge_id="wrong-id",
            prover_fingerprint=response.prover_fingerprint,
            solution_nonce=response.solution_nonce,
            solution_hash=response.solution_hash,
            solved_at=response.solved_at,
            elapsed_seconds=response.elapsed_seconds,
        )
        assert not poc.verify_response(challenge, tampered)

    def test_verify_expired_challenge_fails(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        # Create an expired version of the challenge
        expired = ComputeChallenge(
            challenge_id=challenge.challenge_id,
            challenge_nonce=challenge.challenge_nonce,
            difficulty=challenge.difficulty,
            challenger_fingerprint=challenge.challenger_fingerprint,
            issued_at=challenge.issued_at - 400,
            expires_at=challenge.issued_at - 100,
        )
        assert not poc.verify_response(expired, response)

    def test_solve_expired_challenge_returns_none(self) -> None:
        poc = self._poc(difficulty=1)
        expired = ComputeChallenge(
            challenge_id="exp",
            challenge_nonce="ff",
            difficulty=1,
            challenger_fingerprint="a" * 64,
            issued_at=time.time() - 400,
            expires_at=time.time() - 1,
        )
        assert poc.solve_challenge(expired) is None

    def test_response_elapsed_positive(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        assert response.elapsed_seconds >= 0

    def test_response_to_dict(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        d = response.to_dict()
        assert "solution_nonce" in d
        assert "prover_fingerprint" in d
        assert "elapsed_seconds" in d

    def test_challenger_fingerprint_in_challenge(self) -> None:
        poc = self._poc()
        challenge = poc.issue_challenge()
        assert challenge.challenger_fingerprint == poc.attestor.profile().fingerprint

    def test_prover_fingerprint_in_response(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        response = poc.solve_challenge(challenge)
        assert response is not None
        assert response.prover_fingerprint == poc.attestor.profile().fingerprint

    def test_custom_prover_fingerprint(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        custom_fp = "b" * 64
        response = poc.solve_challenge(challenge, prover_fingerprint=custom_fp)
        assert response is not None
        assert response.prover_fingerprint == custom_fp

    def test_verify_with_custom_prover_fingerprint(self) -> None:
        poc = self._poc(difficulty=1)
        challenge = poc.issue_challenge(difficulty=1)
        custom_fp = "c" * 64
        response = poc.solve_challenge(challenge, prover_fingerprint=custom_fp)
        assert response is not None
        assert poc.verify_response(challenge, response)

    def test_different_challenges_have_different_nonces(self) -> None:
        poc = self._poc()
        c1 = poc.issue_challenge()
        c2 = poc.issue_challenge()
        assert c1.challenge_nonce != c2.challenge_nonce or c1.challenge_id != c2.challenge_id
