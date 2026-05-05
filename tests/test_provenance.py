"""Tests for Phase 29: Decision Provenance (manifold/provenance.py)."""

from __future__ import annotations

import hashlib
import time

import pytest

from manifold.provenance import (
    DecisionReceipt,
    ProvenanceLedger,
    _GENESIS_HASH,
    hash_policy,
    make_task_id,
)


# ---------------------------------------------------------------------------
# DecisionReceipt tests
# ---------------------------------------------------------------------------


class TestDecisionReceipt:
    def _make_receipt(self, **kwargs) -> DecisionReceipt:
        defaults = {
            "timestamp": 1_000_000.0,
            "task_id": "task-001",
            "grid_state_summary": {"action": "use_tool", "risk_score": 0.2},
            "braintrust_votes": ({"genome": "balanced", "approves": True},),
            "policy_hash": "abc123",
            "final_decision": "use_tool",
            "previous_hash": _GENESIS_HASH,
        }
        defaults.update(kwargs)
        return DecisionReceipt(**defaults)

    def test_receipt_is_frozen(self) -> None:
        r = self._make_receipt()
        with pytest.raises((AttributeError, TypeError)):
            r.task_id = "modified"  # type: ignore[misc]

    def test_receipt_hash_is_sha256_hex(self) -> None:
        r = self._make_receipt()
        h = r.receipt_hash
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self) -> None:
        r1 = self._make_receipt()
        r2 = self._make_receipt()
        assert r1.receipt_hash == r2.receipt_hash

    def test_different_content_different_hash(self) -> None:
        r1 = self._make_receipt(final_decision="use_tool")
        r2 = self._make_receipt(final_decision="refuse")
        assert r1.receipt_hash != r2.receipt_hash

    def test_to_dict_contains_required_keys(self) -> None:
        r = self._make_receipt()
        d = r.to_dict()
        for key in (
            "timestamp", "task_id", "grid_state_summary", "braintrust_votes",
            "policy_hash", "final_decision", "previous_hash", "receipt_hash",
        ):
            assert key in d

    def test_to_dict_receipt_hash_matches_property(self) -> None:
        r = self._make_receipt()
        assert r.to_dict()["receipt_hash"] == r.receipt_hash

    def test_to_dict_braintrust_votes_is_list(self) -> None:
        r = self._make_receipt()
        assert isinstance(r.to_dict()["braintrust_votes"], list)

    def test_empty_braintrust_votes(self) -> None:
        r = self._make_receipt(braintrust_votes=())
        assert r.to_dict()["braintrust_votes"] == []

    def test_grid_state_summary_dict(self) -> None:
        r = self._make_receipt(grid_state_summary={"action": "plan", "risk_score": 0.9})
        assert r.grid_state_summary["action"] == "plan"

    def test_policy_hash_empty_string(self) -> None:
        r = self._make_receipt(policy_hash="")
        assert r.policy_hash == ""
        h = r.receipt_hash
        assert len(h) == 64

    def test_hash_changes_with_previous_hash(self) -> None:
        r1 = self._make_receipt(previous_hash=_GENESIS_HASH)
        r2 = self._make_receipt(previous_hash="different_prev_hash")
        assert r1.receipt_hash != r2.receipt_hash


# ---------------------------------------------------------------------------
# ProvenanceLedger tests
# ---------------------------------------------------------------------------


class TestProvenanceLedger:
    def test_empty_ledger_receipt_count(self) -> None:
        ledger = ProvenanceLedger()
        assert ledger.receipt_count() == 0

    def test_empty_ledger_head_hash_is_genesis(self) -> None:
        ledger = ProvenanceLedger()
        assert ledger.head_hash() == _GENESIS_HASH

    def test_record_returns_receipt(self) -> None:
        ledger = ProvenanceLedger()
        r = ledger.record("task-1", "use_tool")
        assert isinstance(r, DecisionReceipt)

    def test_record_increments_count(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("task-1", "use_tool")
        assert ledger.receipt_count() == 1
        ledger.record("task-2", "refuse")
        assert ledger.receipt_count() == 2

    def test_get_returns_receipt_by_task_id(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("task-1", "use_tool")
        r = ledger.get("task-1")
        assert r is not None
        assert r.final_decision == "use_tool"

    def test_get_returns_none_for_unknown_task(self) -> None:
        ledger = ProvenanceLedger()
        assert ledger.get("nonexistent") is None

    def test_first_receipt_previous_hash_is_genesis(self) -> None:
        ledger = ProvenanceLedger()
        r = ledger.record("t1", "plan")
        assert r.previous_hash == _GENESIS_HASH

    def test_second_receipt_previous_hash_is_first_receipt_hash(self) -> None:
        ledger = ProvenanceLedger()
        r1 = ledger.record("t1", "plan")
        r2 = ledger.record("t2", "use_tool")
        assert r2.previous_hash == r1.receipt_hash

    def test_verify_chain_empty(self) -> None:
        ledger = ProvenanceLedger()
        assert ledger.verify_chain() is True

    def test_verify_chain_single(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("t1", "use_tool")
        assert ledger.verify_chain() is True

    def test_verify_chain_multiple(self) -> None:
        ledger = ProvenanceLedger()
        for i in range(5):
            ledger.record(f"task-{i}", "use_tool")
        assert ledger.verify_chain() is True

    def test_verify_chain_detects_tampering(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("t1", "use_tool")
        ledger.record("t2", "plan")
        # Tamper: inject a receipt with a broken previous_hash
        bad_receipt = DecisionReceipt(
            timestamp=time.time(),
            task_id="t3",
            grid_state_summary={},
            braintrust_votes=(),
            policy_hash="",
            final_decision="refuse",
            previous_hash="TAMPERED_HASH",
        )
        ledger._receipts.append(bad_receipt)
        assert ledger.verify_chain() is False

    def test_all_receipts_returns_list(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("t1", "plan")
        ledger.record("t2", "use_tool")
        receipts = ledger.all_receipts()
        assert len(receipts) == 2

    def test_head_hash_after_records(self) -> None:
        ledger = ProvenanceLedger()
        r = ledger.record("t1", "use_tool")
        assert ledger.head_hash() == r.receipt_hash

    def test_record_with_braintrust_votes(self) -> None:
        ledger = ProvenanceLedger()
        votes: tuple[dict, ...] = (
            {"genome": "balanced", "approves": True},
            {"genome": "risk_averse", "approves": False},
        )
        r = ledger.record("t1", "use_tool", braintrust_votes=votes)
        assert len(r.braintrust_votes) == 2

    def test_record_with_policy_hash(self) -> None:
        ledger = ProvenanceLedger()
        r = ledger.record("t1", "use_tool", policy_hash="deadbeef")
        assert r.policy_hash == "deadbeef"

    def test_record_with_custom_clock(self) -> None:
        ledger = ProvenanceLedger()
        fixed_ts = 9_999_999.0
        r = ledger.record("t1", "use_tool", clock=lambda: fixed_ts)
        assert r.timestamp == fixed_ts

    def test_duplicate_task_id_last_write_wins(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("t1", "use_tool")
        ledger.record("t1", "refuse")
        assert ledger.get("t1") is not None
        assert ledger.get("t1").final_decision == "refuse"  # type: ignore[union-attr]

    def test_chain_grows_with_correct_links(self) -> None:
        ledger = ProvenanceLedger()
        receipts = [ledger.record(f"t{i}", "plan") for i in range(10)]
        assert ledger.verify_chain() is True
        assert ledger.receipt_count() == 10
        # Verify linkage explicitly
        assert receipts[0].previous_hash == _GENESIS_HASH
        for i in range(1, 10):
            assert receipts[i].previous_hash == receipts[i - 1].receipt_hash


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_make_task_id_is_16_chars(self) -> None:
        tid = make_task_id("hello world", "finance")
        assert len(tid) == 16

    def test_make_task_id_deterministic(self) -> None:
        tid1 = make_task_id("same prompt", "general")
        tid2 = make_task_id("same prompt", "general")
        assert tid1 == tid2

    def test_make_task_id_different_prompts(self) -> None:
        tid1 = make_task_id("prompt A", "general")
        tid2 = make_task_id("prompt B", "general")
        assert tid1 != tid2

    def test_hash_policy_returns_64_hex(self) -> None:
        class FakePolicy:
            def to_dict(self):
                return {"max_risk": 0.5, "domain": "finance"}

        h = hash_policy(FakePolicy())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_policy_deterministic(self) -> None:
        class FakePolicy:
            def to_dict(self):
                return {"max_risk": 0.5}

        assert hash_policy(FakePolicy()) == hash_policy(FakePolicy())

    def test_hash_policy_fallback_for_no_to_dict(self) -> None:
        h = hash_policy("plain string policy")
        assert len(h) == 64

    def test_genesis_hash_is_sha256(self) -> None:
        expected = hashlib.sha256(b"MANIFOLD_PROVENANCE_GENESIS_v1").hexdigest()
        assert _GENESIS_HASH == expected
