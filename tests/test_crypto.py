"""Tests for Phase 21: Zero-Trust Cryptography."""

from __future__ import annotations

import pytest

from manifold.brain import GossipNote
from manifold.b2b import OrgPolicy
from manifold.policy import ManifoldPolicy, PolicyDomain
from manifold.crypto import (
    GossipSigner,
    OrgPolicySigner,
    PolicySigningKey,
    SignatureVerificationError,
    SignedGossipNote,
    SignedOrgPolicy,
    VerifiedPolicyHandshake,
    _canonical_bytes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(key_id: str = "test-key") -> PolicySigningKey:
    return PolicySigningKey.from_passphrase("secret-passphrase", key_id=key_id)


def _remote_policy(org_id: str = "org-b") -> OrgPolicy:
    return OrgPolicy(
        org_id=org_id,
        min_reliability=0.85,
        max_risk=0.25,
        domain="finance",
    )


def _local_policy() -> ManifoldPolicy:
    d = PolicyDomain(
        name="finance",
        stakes=0.9,
        risk_tolerance=0.30,
        min_tool_reliability=0.80,
    )
    return ManifoldPolicy(domains=[d])


def _gossip_note() -> GossipNote:
    return GossipNote(
        tool="wolfram-alpha",
        claim="healthy",
        source_id="agent-x",
        source_reputation=0.90,
    )


# ---------------------------------------------------------------------------
# PolicySigningKey
# ---------------------------------------------------------------------------


class TestPolicySigningKey:
    def test_generate_returns_unique_keys(self) -> None:
        k1 = PolicySigningKey.generate("k1")
        k2 = PolicySigningKey.generate("k2")
        assert k1.secret != k2.secret

    def test_generate_key_id_set(self) -> None:
        k = PolicySigningKey.generate("my-key")
        assert k.key_id == "my-key"

    def test_generate_length(self) -> None:
        k = PolicySigningKey.generate(length=64)
        assert len(k.secret) == 64

    def test_from_passphrase_deterministic(self) -> None:
        k1 = PolicySigningKey.from_passphrase("hello", "k1")
        k2 = PolicySigningKey.from_passphrase("hello", "k2")
        assert k1.secret == k2.secret

    def test_from_passphrase_different_passphrases(self) -> None:
        k1 = PolicySigningKey.from_passphrase("hello")
        k2 = PolicySigningKey.from_passphrase("world")
        assert k1.secret != k2.secret

    def test_sign_returns_hex_string(self) -> None:
        k = _key()
        sig = k.sign(b"data")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex is 64 chars

    def test_verify_correct_signature(self) -> None:
        k = _key()
        sig = k.sign(b"hello world")
        assert k.verify(b"hello world", sig)

    def test_verify_wrong_data(self) -> None:
        k = _key()
        sig = k.sign(b"correct")
        assert not k.verify(b"wrong", sig)

    def test_verify_tampered_signature(self) -> None:
        k = _key()
        sig = k.sign(b"data")
        tampered = sig[:-1] + ("0" if sig[-1] != "0" else "1")
        assert not k.verify(b"data", tampered)

    def test_different_keys_dont_verify_each_other(self) -> None:
        k1 = _key("k1")
        k2 = PolicySigningKey.from_passphrase("different", "k2")
        sig = k1.sign(b"data")
        assert not k2.verify(b"data", sig)


# ---------------------------------------------------------------------------
# _canonical_bytes
# ---------------------------------------------------------------------------


class TestCanonicalBytes:
    def test_sorted_keys(self) -> None:
        b1 = _canonical_bytes({"b": 1, "a": 2})
        b2 = _canonical_bytes({"a": 2, "b": 1})
        assert b1 == b2

    def test_is_bytes(self) -> None:
        result = _canonical_bytes({"x": 1})
        assert isinstance(result, bytes)

    def test_no_whitespace(self) -> None:
        result = _canonical_bytes({"a": 1})
        assert b" " not in result


# ---------------------------------------------------------------------------
# OrgPolicySigner
# ---------------------------------------------------------------------------


class TestOrgPolicySigner:
    def test_sign_returns_signed_org_policy(self) -> None:
        signer = OrgPolicySigner(_key())
        policy = _remote_policy()
        signed = signer.sign(policy)
        assert isinstance(signed, SignedOrgPolicy)
        assert signed.policy == policy
        assert isinstance(signed.signature, str)
        assert signed.key_id == "test-key"

    def test_verify_valid_signature(self) -> None:
        key = _key()
        signer = OrgPolicySigner(key)
        policy = _remote_policy()
        signed = signer.sign(policy)
        result = signer.verify(signed)
        assert result == policy

    def test_verify_raises_on_tampered_signature(self) -> None:
        signer = OrgPolicySigner(_key())
        policy = _remote_policy()
        signed = signer.sign(policy)
        tampered_sig = signed.signature[:-1] + ("0" if signed.signature[-1] != "0" else "1")
        bad_signed = SignedOrgPolicy(
            policy=policy, signature=tampered_sig, key_id=signed.key_id
        )
        with pytest.raises(SignatureVerificationError):
            signer.verify(bad_signed)

    def test_verify_raises_on_tampered_policy(self) -> None:
        signer = OrgPolicySigner(_key())
        policy = _remote_policy()
        signed = signer.sign(policy)
        tampered_policy = OrgPolicy(
            org_id=policy.org_id,
            min_reliability=0.99,  # tampered
            max_risk=policy.max_risk,
            domain=policy.domain,
        )
        bad_signed = SignedOrgPolicy(
            policy=tampered_policy, signature=signed.signature, key_id=signed.key_id
        )
        with pytest.raises(SignatureVerificationError):
            signer.verify(bad_signed)

    def test_verify_bool_true(self) -> None:
        signer = OrgPolicySigner(_key())
        policy = _remote_policy()
        signed = signer.sign(policy)
        assert signer.verify_bool(signed) is True

    def test_verify_bool_false(self) -> None:
        signer = OrgPolicySigner(_key())
        policy = _remote_policy()
        signed = signer.sign(policy)
        bad = SignedOrgPolicy(policy=policy, signature="bad", key_id=signed.key_id)
        assert signer.verify_bool(bad) is False

    def test_round_trip_to_dict(self) -> None:
        signer = OrgPolicySigner(_key())
        policy = _remote_policy()
        signed = signer.sign(policy)
        d = signed.to_dict()
        restored = SignedOrgPolicy.from_dict(d)
        assert restored.policy == signed.policy
        assert restored.signature == signed.signature
        assert restored.key_id == signed.key_id

    def test_different_key_cannot_verify(self) -> None:
        signer1 = OrgPolicySigner(PolicySigningKey.from_passphrase("key1", "k1"))
        signer2 = OrgPolicySigner(PolicySigningKey.from_passphrase("key2", "k2"))
        policy = _remote_policy()
        signed = signer1.sign(policy)
        assert not signer2.verify_bool(signed)


# ---------------------------------------------------------------------------
# GossipSigner
# ---------------------------------------------------------------------------


class TestGossipSigner:
    def test_sign_returns_signed_gossip_note(self) -> None:
        signer = GossipSigner(_key())
        note = _gossip_note()
        signed = signer.sign(note)
        assert isinstance(signed, SignedGossipNote)
        assert signed.note == note
        assert len(signed.signature) == 64

    def test_verify_valid(self) -> None:
        signer = GossipSigner(_key())
        note = _gossip_note()
        signed = signer.sign(note)
        result = signer.verify(signed)
        assert result == note

    def test_verify_raises_on_tampered_note(self) -> None:
        signer = GossipSigner(_key())
        note = _gossip_note()
        signed = signer.sign(note)
        tampered_note = GossipNote(
            tool="wolfram-alpha",
            claim="failing",  # tampered from "healthy"
            source_id="agent-x",
            source_reputation=0.90,
        )
        bad = SignedGossipNote(
            note=tampered_note, signature=signed.signature, key_id=signed.key_id
        )
        with pytest.raises(SignatureVerificationError):
            signer.verify(bad)

    def test_verify_bool_true(self) -> None:
        signer = GossipSigner(_key())
        note = _gossip_note()
        signed = signer.sign(note)
        assert signer.verify_bool(signed) is True

    def test_verify_bool_false_bad_sig(self) -> None:
        signer = GossipSigner(_key())
        note = _gossip_note()
        signed = signer.sign(note)
        bad = SignedGossipNote(note=note, signature="bad_sig", key_id=signed.key_id)
        assert signer.verify_bool(bad) is False

    def test_signed_gossip_to_dict(self) -> None:
        signer = GossipSigner(_key())
        note = _gossip_note()
        signed = signer.sign(note)
        d = signed.to_dict()
        assert d["key_id"] == "test-key"
        assert "signature" in d
        assert d["note"]["tool"] == "wolfram-alpha"

    def test_different_note_different_sig(self) -> None:
        signer = GossipSigner(_key())
        note1 = _gossip_note()
        note2 = GossipNote(
            tool="gpt-4",
            claim="failing",
            source_id="agent-y",
            source_reputation=0.50,
        )
        s1 = signer.sign(note1)
        s2 = signer.sign(note2)
        assert s1.signature != s2.signature


# ---------------------------------------------------------------------------
# SignatureVerificationError
# ---------------------------------------------------------------------------


class TestSignatureVerificationError:
    def test_is_value_error(self) -> None:
        err = SignatureVerificationError("test reason", key_id="k1")
        assert isinstance(err, ValueError)

    def test_attributes(self) -> None:
        err = SignatureVerificationError("reason here", key_id="k42")
        assert err.reason == "reason here"
        assert err.key_id == "k42"

    def test_default_key_id(self) -> None:
        err = SignatureVerificationError("oops")
        assert err.key_id == ""


# ---------------------------------------------------------------------------
# VerifiedPolicyHandshake
# ---------------------------------------------------------------------------


class TestVerifiedPolicyHandshake:
    def test_check_signed_valid(self) -> None:
        key = _key("org-b-key")
        signer = OrgPolicySigner(key)
        remote = _remote_policy()
        signed = signer.sign(remote)

        vhs = VerifiedPolicyHandshake(
            local_policy=_local_policy(),
            local_org_id="org-a",
            trusted_keys={"org-b-key": key},
        )
        result = vhs.check_signed(signed)
        assert result.compatible is True

    def test_check_signed_unknown_key_raises(self) -> None:
        key = _key("org-b-key")
        signer = OrgPolicySigner(key)
        remote = _remote_policy()
        signed = signer.sign(remote)

        vhs = VerifiedPolicyHandshake(
            local_policy=_local_policy(),
            local_org_id="org-a",
            trusted_keys={},  # no trusted keys
        )
        with pytest.raises(SignatureVerificationError) as exc_info:
            vhs.check_signed(signed)
        assert "Unknown key_id" in str(exc_info.value)

    def test_check_signed_tampered_raises(self) -> None:
        key = _key("org-b-key")
        signer = OrgPolicySigner(key)
        remote = _remote_policy()
        signed = signer.sign(remote)
        tampered = SignedOrgPolicy(
            policy=OrgPolicy(
                org_id=remote.org_id,
                min_reliability=0.10,  # tampered
                max_risk=remote.max_risk,
                domain=remote.domain,
            ),
            signature=signed.signature,
            key_id=signed.key_id,
        )

        vhs = VerifiedPolicyHandshake(
            local_policy=_local_policy(),
            local_org_id="org-a",
            trusted_keys={"org-b-key": key},
        )
        with pytest.raises(SignatureVerificationError):
            vhs.check_signed(tampered)

    def test_check_signed_incompatible_policies(self) -> None:
        key = _key("org-b-key")
        signer = OrgPolicySigner(key)
        # Remote has max_risk=0.90, far exceeding local tolerance=0.30
        remote = OrgPolicy(
            org_id="org-b",
            min_reliability=0.50,
            max_risk=0.90,
            domain="finance",
        )
        signed = signer.sign(remote)

        vhs = VerifiedPolicyHandshake(
            local_policy=_local_policy(),
            local_org_id="org-a",
            trusted_keys={"org-b-key": key},
        )
        result = vhs.check_signed(signed)
        assert result.compatible is False

    def test_add_trusted_key(self) -> None:
        vhs = VerifiedPolicyHandshake(
            local_policy=_local_policy(), local_org_id="org-a"
        )
        assert not vhs.trusted_keys
        key = _key("new-key")
        vhs.add_trusted_key(key)
        assert "new-key" in vhs.trusted_keys

    def test_inherits_policy_handshake(self) -> None:
        from manifold.b2b import PolicyHandshake
        vhs = VerifiedPolicyHandshake(local_policy=_local_policy(), local_org_id="a")
        assert isinstance(vhs, PolicyHandshake)
