"""Tests for Phase 30: Trust-Based Rate Limiting (manifold/quota.py)."""

from __future__ import annotations

import time

import pytest

from manifold.quota import QuotaExhaustedError, QuotaManager, TrustTokenBucket


# ---------------------------------------------------------------------------
# QuotaExhaustedError tests
# ---------------------------------------------------------------------------


class TestQuotaExhaustedError:
    def test_is_runtime_error(self) -> None:
        exc = QuotaExhaustedError("tool-x", 0.5, 10.0)
        assert isinstance(exc, RuntimeError)

    def test_attributes_set(self) -> None:
        exc = QuotaExhaustedError("tool-x", 0.5, 10.0)
        assert exc.entity_id == "tool-x"
        assert exc.available == 0.5
        assert exc.capacity == 10.0

    def test_message_contains_entity_id(self) -> None:
        exc = QuotaExhaustedError("my-tool", 0.0, 5.0)
        assert "my-tool" in str(exc)

    def test_zero_available(self) -> None:
        exc = QuotaExhaustedError("t", 0.0, 10.0)
        assert exc.available == 0.0


# ---------------------------------------------------------------------------
# TrustTokenBucket tests
# ---------------------------------------------------------------------------


class TestTrustTokenBucket:
    def test_initial_tokens_equals_capacity(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=8.0)
        assert b.tokens == pytest.approx(8.0)

    def test_consume_reduces_tokens(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=10.0)
        assert b.consume(3.0) is True
        assert b.tokens == pytest.approx(7.0)

    def test_consume_exact_capacity(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=5.0)
        assert b.consume(5.0) is True
        assert b.tokens == pytest.approx(0.0)

    def test_consume_more_than_available_returns_false(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=5.0)
        b.consume(5.0)  # drain
        assert b.consume(1.0) is False

    def test_consume_does_not_go_negative(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=2.0)
        b.consume(2.0)
        b.consume(1.0)  # fails
        assert b.tokens == pytest.approx(0.0)

    def test_refill_adds_tokens_up_to_capacity(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=10.0, refill_rate=100.0)
        b.consume(10.0)  # drain
        # Simulate 1 second passing
        b._last_refill -= 1.0
        b.refill(trust_score=1.0, entropy=0.0)
        assert b.tokens == pytest.approx(10.0)

    def test_refill_does_not_exceed_capacity(self) -> None:
        b = TrustTokenBucket(entity_id="tool-a", capacity=5.0, refill_rate=1000.0)
        b._last_refill -= 10.0  # 10 seconds of refill
        b.refill(trust_score=1.0, entropy=0.0)
        assert b.tokens <= 5.0

    def test_refill_penalised_by_low_trust(self) -> None:
        b1 = TrustTokenBucket(entity_id="full-trust", capacity=100.0, refill_rate=1.0)
        b2 = TrustTokenBucket(entity_id="low-trust", capacity=100.0, refill_rate=1.0)
        b1.consume(100.0)
        b2.consume(100.0)
        b1._last_refill -= 10.0
        b2._last_refill -= 10.0
        b1.refill(trust_score=1.0, entropy=0.0)
        b2.refill(trust_score=0.1, entropy=0.0)
        assert b1.tokens > b2.tokens

    def test_refill_penalised_by_high_entropy(self) -> None:
        b1 = TrustTokenBucket(entity_id="low-entropy", capacity=100.0, refill_rate=1.0)
        b2 = TrustTokenBucket(entity_id="high-entropy", capacity=100.0, refill_rate=1.0)
        b1.consume(100.0)
        b2.consume(100.0)
        b1._last_refill -= 10.0
        b2._last_refill -= 10.0
        b1.refill(trust_score=1.0, entropy=0.0)
        b2.refill(trust_score=1.0, entropy=0.99)
        assert b1.tokens > b2.tokens

    def test_minimum_refill_rate_is_nonzero(self) -> None:
        b = TrustTokenBucket(entity_id="dead", capacity=100.0, refill_rate=1.0)
        b.consume(100.0)
        b._last_refill -= 10000.0  # very long wait
        b.refill(trust_score=0.0, entropy=1.0)
        # Should still gain some tokens (min rate = 0.01)
        assert b.tokens > 0.0

    def test_available_tokens_matches_tokens(self) -> None:
        b = TrustTokenBucket(entity_id="t", capacity=5.0)
        b.consume(2.0)
        assert b.available_tokens() == pytest.approx(3.0)

    def test_utilisation_full(self) -> None:
        b = TrustTokenBucket(entity_id="t", capacity=10.0)
        assert b.utilisation() == pytest.approx(0.0)

    def test_utilisation_empty(self) -> None:
        b = TrustTokenBucket(entity_id="t", capacity=10.0)
        b.consume(10.0)
        assert b.utilisation() == pytest.approx(1.0)

    def test_utilisation_partial(self) -> None:
        b = TrustTokenBucket(entity_id="t", capacity=10.0)
        b.consume(5.0)
        assert b.utilisation() == pytest.approx(0.5)

    def test_effective_refill_rate_clamped(self) -> None:
        b = TrustTokenBucket(entity_id="t", capacity=10.0, refill_rate=1.0)
        # extreme values should not cause errors
        rate = b._effective_refill_rate(0.0, 1.0)
        assert rate >= 0.01
        rate2 = b._effective_refill_rate(1.0, 0.0)
        assert rate2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# QuotaManager tests
# ---------------------------------------------------------------------------


class TestQuotaManager:
    def test_register_returns_bucket(self) -> None:
        qm = QuotaManager()
        b = qm.register("tool-x")
        assert isinstance(b, TrustTokenBucket)

    def test_register_idempotent(self) -> None:
        qm = QuotaManager()
        b1 = qm.register("tool-x")
        b2 = qm.register("tool-x")
        assert b1 is b2

    def test_probationary_bucket_has_smaller_capacity(self) -> None:
        qm = QuotaManager(default_capacity=10.0, probationary_burst_limit=3.0)
        b = qm.register("prob-tool", probationary=True)
        assert b.capacity == pytest.approx(3.0)

    def test_non_probationary_bucket_has_default_capacity(self) -> None:
        qm = QuotaManager(default_capacity=10.0)
        b = qm.register("normal-tool")
        assert b.capacity == pytest.approx(10.0)

    def test_check_and_consume_succeeds_with_tokens(self) -> None:
        qm = QuotaManager()
        qm.register("tool-a")
        # Should not raise
        qm.check_and_consume("tool-a")

    def test_check_and_consume_raises_when_empty(self) -> None:
        qm = QuotaManager(default_capacity=1.0)
        qm.register("tool-x")
        qm.check_and_consume("tool-x")  # consumes the 1 token
        # Drain refill time so no tokens are added
        bucket = qm.bucket("tool-x")
        assert bucket is not None
        bucket._last_refill = time.time() + 9999  # trick: no time has passed
        bucket._tokens = 0.0
        with pytest.raises(QuotaExhaustedError) as exc_info:
            qm.check_and_consume("tool-x")
        assert exc_info.value.entity_id == "tool-x"

    def test_auto_creates_bucket_on_consume(self) -> None:
        qm = QuotaManager(default_capacity=10.0)
        # entity was never explicitly registered
        qm.check_and_consume("new-tool")
        assert qm.bucket("new-tool") is not None

    def test_is_probationary(self) -> None:
        qm = QuotaManager()
        qm.register("prob-tool", probationary=True)
        qm.register("normal-tool")
        assert qm.is_probationary("prob-tool") is True
        assert qm.is_probationary("normal-tool") is False

    def test_mark_probationary(self) -> None:
        qm = QuotaManager(default_capacity=10.0, probationary_burst_limit=2.0)
        qm.register("t")
        assert not qm.is_probationary("t")
        qm.mark_probationary("t")
        assert qm.is_probationary("t")
        assert qm.bucket("t").capacity == pytest.approx(2.0)  # type: ignore[union-attr]

    def test_bucket_returns_none_for_unknown(self) -> None:
        qm = QuotaManager()
        assert qm.bucket("never-registered") is None

    def test_quota_summary_structure(self) -> None:
        qm = QuotaManager(default_capacity=10.0)
        qm.register("t1")
        qm.register("t2")
        summary = qm.quota_summary()
        assert "t1" in summary
        assert "tokens" in summary["t1"]
        assert "capacity" in summary["t1"]
        assert "utilisation" in summary["t1"]

    def test_all_entity_ids_sorted(self) -> None:
        qm = QuotaManager()
        for name in ["zzz", "aaa", "mmm"]:
            qm.register(name)
        ids = qm.all_entity_ids()
        assert ids == sorted(ids)

    def test_custom_capacity_override(self) -> None:
        qm = QuotaManager(default_capacity=10.0)
        b = qm.register("t", capacity=25.0)
        assert b.capacity == pytest.approx(25.0)

    def test_custom_refill_rate_override(self) -> None:
        qm = QuotaManager(default_refill_rate=1.0)
        b = qm.register("t", refill_rate=5.0)
        assert b.refill_rate == pytest.approx(5.0)

    def test_consume_multiple_tokens(self) -> None:
        qm = QuotaManager(default_capacity=20.0)
        qm.register("t")
        qm.check_and_consume("t", amount=5.0)
        b = qm.bucket("t")
        assert b is not None
        assert b.tokens <= 15.0

    def test_with_hub_none_defaults(self) -> None:
        qm = QuotaManager(hub=None)
        qm.register("tool")
        # Should not crash — defaults trust=1.0, entropy=0.0
        qm.check_and_consume("tool")
