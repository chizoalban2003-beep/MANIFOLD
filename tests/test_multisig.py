"""Tests for Phase 40: Multi-Sig Execution (manifold/multisig.py)."""

from __future__ import annotations

import time

import pytest

from manifold.brain import BrainTask
from manifold.crypto import PolicySigningKey
from manifold.multisig import (
    MultiSigConfig,
    MultiSigVault,
    PeerEndorsement,
    sign_endorsement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _high_task(prompt: str = "high-stakes-op") -> BrainTask:
    return BrainTask(prompt=prompt, domain="finance", stakes=0.95)


def _low_task(prompt: str = "low-stakes-op") -> BrainTask:
    return BrainTask(prompt=prompt, domain="general", stakes=0.3)


def _vault_with_keys(
    n: int = 3,
    m: int = 2,
) -> tuple[MultiSigVault, list[PolicySigningKey]]:
    vault = MultiSigVault(config=MultiSigConfig(required_signatures=m, total_peers=n))
    keys = [PolicySigningKey.generate(f"peer-{i}") for i in range(n)]
    for k in keys:
        vault.add_trusted_key(k)
    return vault, keys


# ---------------------------------------------------------------------------
# MultiSigConfig
# ---------------------------------------------------------------------------


class TestMultiSigConfig:
    def test_defaults(self) -> None:
        cfg = MultiSigConfig()
        assert cfg.required_signatures == 2
        assert cfg.total_peers == 3
        assert cfg.high_stakes_threshold == 0.9
        assert cfg.signature_ttl_seconds == 300.0

    def test_custom(self) -> None:
        cfg = MultiSigConfig(required_signatures=3, total_peers=5, high_stakes_threshold=0.8)
        assert cfg.required_signatures == 3
        assert cfg.total_peers == 5
        assert cfg.high_stakes_threshold == 0.8

    def test_frozen(self) -> None:
        cfg = MultiSigConfig()
        with pytest.raises(Exception):
            cfg.required_signatures = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PeerEndorsement helpers
# ---------------------------------------------------------------------------


class TestSignEndorsement:
    def test_creates_endorsement(self) -> None:
        key = PolicySigningKey.generate("p1")
        task = _high_task()
        endorsement = sign_endorsement(task, "org-x", key)
        assert isinstance(endorsement, PeerEndorsement)
        assert endorsement.peer_org_id == "org-x"
        assert endorsement.key_id == "p1"
        assert len(endorsement.task_hash) == 64  # SHA-256 hex

    def test_signature_verifiable(self) -> None:
        key = PolicySigningKey.generate("k1")
        task = _high_task()
        e = sign_endorsement(task, "org-a", key)
        assert key.verify(
            __import__("manifold.multisig", fromlist=["_task_canonical"])._task_canonical(task),
            e.signature,
        )

    def test_different_tasks_different_hash(self) -> None:
        key = PolicySigningKey.generate("k2")
        t1 = BrainTask(prompt="task1", stakes=0.95)
        t2 = BrainTask(prompt="task2", stakes=0.95)
        e1 = sign_endorsement(t1, "org-a", key)
        e2 = sign_endorsement(t2, "org-a", key)
        assert e1.task_hash != e2.task_hash

    def test_to_dict(self) -> None:
        key = PolicySigningKey.generate("k3")
        e = sign_endorsement(_high_task(), "org-b", key)
        d = e.to_dict()
        assert "peer_org_id" in d
        assert "task_hash" in d
        assert "signature" in d
        assert "key_id" in d


# ---------------------------------------------------------------------------
# MultiSigVault.submit
# ---------------------------------------------------------------------------


class TestMultiSigVaultSubmit:
    def test_submit_returns_entry_id(self) -> None:
        vault, _ = _vault_with_keys()
        entry_id = vault.submit(_high_task())
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_submit_low_stakes_raises(self) -> None:
        vault, _ = _vault_with_keys()
        with pytest.raises(ValueError, match="below"):
            vault.submit(_low_task())

    def test_entry_stored(self) -> None:
        vault, _ = _vault_with_keys()
        entry_id = vault.submit(_high_task())
        entry = vault.get_entry(entry_id)
        assert entry is not None
        assert entry.status == "pending"

    def test_pending_count_increases(self) -> None:
        vault, _ = _vault_with_keys()
        assert vault.pending_count() == 0
        vault.submit(_high_task())
        assert vault.pending_count() == 1

    def test_requires_multisig_threshold(self) -> None:
        vault, _ = _vault_with_keys()
        assert vault.requires_multisig(_high_task()) is True
        assert vault.requires_multisig(_low_task()) is False

    def test_submit_at_exact_threshold(self) -> None:
        vault = MultiSigVault(config=MultiSigConfig(high_stakes_threshold=0.9))
        task = BrainTask(prompt="edge", stakes=0.9)
        entry_id = vault.submit(task)
        assert entry_id is not None


# ---------------------------------------------------------------------------
# MultiSigVault.endorse — happy path
# ---------------------------------------------------------------------------


class TestMultiSigVaultEndorse:
    def test_one_endorsement_not_yet_approved(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=2)
        task = _high_task()
        entry_id = vault.submit(task)
        e = sign_endorsement(task, "peer-0", keys[0])
        result = vault.endorse(entry_id, e)
        assert result.accepted is True
        assert result.approved is False
        assert result.endorsement_count == 1

    def test_m_endorsements_approves(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=2)
        task = _high_task()
        entry_id = vault.submit(task)
        e0 = sign_endorsement(task, "peer-0", keys[0])
        e1 = sign_endorsement(task, "peer-1", keys[1])
        vault.endorse(entry_id, e0)
        result = vault.endorse(entry_id, e1)
        assert result.accepted is True
        assert result.approved is True
        assert result.status == "approved"

    def test_is_approved_after_m_sigs(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=2)
        task = _high_task()
        entry_id = vault.submit(task)
        for i in range(2):
            e = sign_endorsement(task, f"peer-{i}", keys[i])
            vault.endorse(entry_id, e)
        assert vault.is_approved(entry_id) is True

    def test_endorse_unknown_entry(self) -> None:
        vault, keys = _vault_with_keys()
        task = _high_task()
        e = sign_endorsement(task, "peer-0", keys[0])
        result = vault.endorse("nonexistent-id", e)
        assert result.accepted is False
        assert result.status == "not_found"

    def test_duplicate_peer_rejected(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=3)
        task = _high_task()
        entry_id = vault.submit(task)
        e = sign_endorsement(task, "peer-0", keys[0])
        vault.endorse(entry_id, e)
        result = vault.endorse(entry_id, e)
        assert result.accepted is False
        assert "already endorsed" in result.reason

    def test_wrong_task_hash_rejected(self) -> None:
        vault, keys = _vault_with_keys()
        task = _high_task()
        entry_id = vault.submit(task)
        # Build endorsement with wrong hash
        bad_e = PeerEndorsement(
            peer_org_id="peer-0",
            task_hash="0" * 64,
            signature="fake",
            key_id=keys[0].key_id,
            timestamp=time.time(),
        )
        result = vault.endorse(entry_id, bad_e)
        assert result.accepted is False
        assert "mismatch" in result.reason

    def test_unknown_key_rejected(self) -> None:
        vault, keys = _vault_with_keys()
        task = _high_task()
        entry_id = vault.submit(task)
        rogue_key = PolicySigningKey.generate("unknown-key")
        e = sign_endorsement(task, "peer-rogue", rogue_key)
        result = vault.endorse(entry_id, e)
        assert result.accepted is False
        assert "unknown key_id" in result.reason

    def test_bad_signature_rejected(self) -> None:
        vault, keys = _vault_with_keys()
        task = _high_task()
        entry_id = vault.submit(task)
        from manifold.multisig import _task_canonical  # noqa: PLC0415
        task_hash = __import__("hashlib").sha256(_task_canonical(task)).hexdigest()
        bad_e = PeerEndorsement(
            peer_org_id="peer-0",
            task_hash=task_hash,
            signature="badsig",
            key_id=keys[0].key_id,
            timestamp=time.time(),
        )
        result = vault.endorse(entry_id, bad_e)
        assert result.accepted is False
        assert "signature verification failed" in result.reason

    def test_endorse_already_approved(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=2)
        task = _high_task()
        entry_id = vault.submit(task)
        for i in range(2):
            e = sign_endorsement(task, f"peer-{i}", keys[i])
            vault.endorse(entry_id, e)
        # Third endorsement on already-approved entry
        e2 = sign_endorsement(task, "peer-2", keys[2])
        result = vault.endorse(entry_id, e2)
        assert result.accepted is False
        assert result.status == "approved"


# ---------------------------------------------------------------------------
# MultiSigVault — TTL & expiry
# ---------------------------------------------------------------------------


class TestMultiSigVaultExpiry:
    def test_expired_entry_rejected(self) -> None:
        vault = MultiSigVault(
            config=MultiSigConfig(signature_ttl_seconds=0.001, high_stakes_threshold=0.9)
        )
        key = PolicySigningKey.generate("k")
        vault.add_trusted_key(key)
        task = _high_task()
        entry_id = vault.submit(task)
        time.sleep(0.01)  # Wait for TTL to expire
        e = sign_endorsement(task, "peer-0", key)
        result = vault.endorse(entry_id, e)
        assert result.accepted is False
        assert result.status == "expired"

    def test_purge_expired(self) -> None:
        vault = MultiSigVault(
            config=MultiSigConfig(signature_ttl_seconds=0.001, high_stakes_threshold=0.9)
        )
        vault.submit(_high_task("t1"))
        vault.submit(_high_task("t2"))
        time.sleep(0.01)
        expired_count = vault.purge_expired()
        assert expired_count == 2


# ---------------------------------------------------------------------------
# MultiSigVault.summary
# ---------------------------------------------------------------------------


class TestMultiSigVaultSummary:
    def test_summary_empty(self) -> None:
        vault, _ = _vault_with_keys()
        s = vault.summary()
        assert s["total_entries"] == 0
        assert s["required_signatures"] == 2
        assert s["total_peers"] == 3

    def test_summary_with_entries(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=2)
        vault.submit(_high_task("t1"))
        vault.submit(_high_task("t2"))
        s = vault.summary()
        assert s["total_entries"] == 2
        assert s["by_status"].get("pending", 0) == 2

    def test_is_approved_false_nonexistent(self) -> None:
        vault, _ = _vault_with_keys()
        assert vault.is_approved("does-not-exist") is False

    def test_get_entry_none_for_missing(self) -> None:
        vault, _ = _vault_with_keys()
        assert vault.get_entry("nope") is None

    def test_add_trusted_key(self) -> None:
        vault = MultiSigVault()
        key = PolicySigningKey.generate("k-new")
        vault.add_trusted_key(key)
        assert "k-new" in vault.trusted_keys

    def test_1_of_1_approval(self) -> None:
        vault = MultiSigVault(config=MultiSigConfig(required_signatures=1, total_peers=1))
        key = PolicySigningKey.generate("solo")
        vault.add_trusted_key(key)
        task = _high_task()
        entry_id = vault.submit(task)
        e = sign_endorsement(task, "peer-0", key)
        result = vault.endorse(entry_id, e)
        assert result.approved is True
        assert result.endorsement_count == 1

    def test_3_of_3_approval(self) -> None:
        vault, keys = _vault_with_keys(n=3, m=3)
        task = _high_task("3of3")
        entry_id = vault.submit(task)
        for i in range(3):
            e = sign_endorsement(task, f"peer-{i}", keys[i])
            result = vault.endorse(entry_id, e)
        assert result.approved is True
        assert vault.is_approved(entry_id) is True

    def test_purge_expired_updates_status(self) -> None:
        vault = MultiSigVault(
            config=MultiSigConfig(signature_ttl_seconds=0.001, high_stakes_threshold=0.9)
        )
        vault.submit(_high_task())
        time.sleep(0.01)
        vault.purge_expired()
        entry = next(iter(vault._entries.values()))  # noqa: SLF001
        assert entry.status == "expired"
