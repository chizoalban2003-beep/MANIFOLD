"""Tests for Phase 26: Reputation Decay & Entropy (manifold/entropy.py)."""

from __future__ import annotations

import math

import pytest

from manifold.entropy import ReputationDecay, VolatilityTable, _DEFAULT_LAMBDA


# ---------------------------------------------------------------------------
# VolatilityTable tests
# ---------------------------------------------------------------------------


class TestVolatilityTable:
    def test_default_factory(self) -> None:
        vt = VolatilityTable.default()
        assert isinstance(vt.lambdas, dict)
        assert len(vt.lambdas) > 0

    def test_known_domain_lambdas(self) -> None:
        vt = VolatilityTable.default()
        # llm decays faster than math
        assert vt.lambda_for("llm") > vt.lambda_for("math")
        assert vt.lambda_for("storage") < vt.lambda_for("search")

    def test_unknown_domain_returns_default(self) -> None:
        vt = VolatilityTable.default()
        assert vt.lambda_for("nonexistent_domain") == vt.default_lambda

    def test_custom_lambda_override(self) -> None:
        vt = VolatilityTable(lambdas={"custom": 0.1}, default_lambda=0.05)
        assert vt.lambda_for("custom") == 0.1
        assert vt.lambda_for("other") == 0.05

    def test_half_life_hours_llm(self) -> None:
        vt = VolatilityTable.default()
        hl = vt.half_life_hours("llm")
        expected = math.log(2) / vt.lambda_for("llm")
        assert abs(hl - expected) < 0.001

    def test_half_life_hours_math_longer_than_llm(self) -> None:
        vt = VolatilityTable.default()
        assert vt.half_life_hours("math") > vt.half_life_hours("llm")

    def test_zero_lambda_returns_infinity(self) -> None:
        vt = VolatilityTable(lambdas={"stable": 0.0}, default_lambda=0.0)
        assert vt.half_life_hours("stable") == float("inf")

    def test_all_default_domains_covered(self) -> None:
        vt = VolatilityTable.default()
        for domain in _DEFAULT_LAMBDA:
            assert vt.lambda_for(domain) == _DEFAULT_LAMBDA[domain]

    def test_lambda_values_positive(self) -> None:
        vt = VolatilityTable.default()
        for domain, lam in vt.lambdas.items():
            assert lam > 0, f"lambda for {domain!r} must be positive"

    def test_frozen_dataclass(self) -> None:
        vt = VolatilityTable.default()
        with pytest.raises((AttributeError, TypeError)):
            vt.default_lambda = 99.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReputationDecay tests
# ---------------------------------------------------------------------------


class TestReputationDecayBasics:
    def _make_decay(self, start_ts: float = 1_000_000.0) -> tuple[ReputationDecay, list[float]]:
        """Return a decay engine with a controllable clock."""
        clock_state = [start_ts]
        decay = ReputationDecay(clock=lambda: clock_state[0])
        return decay, clock_state

    def test_no_signal_returns_base_reliability(self) -> None:
        decay, _ = self._make_decay()
        result = decay.decayed_reliability("tool_x", base_reliability=0.9)
        assert result == pytest.approx(0.9)

    def test_age_hours_none_when_no_signal(self) -> None:
        decay, _ = self._make_decay()
        assert decay.age_hours("tool_x") is None

    def test_record_signal_sets_age_zero(self) -> None:
        decay, clock_state = self._make_decay(start_ts=1_000_000.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.9)
        # Immediately after recording, age is ~0
        age = decay.age_hours("tool_x")
        assert age is not None
        assert age == pytest.approx(0.0, abs=0.01)

    def test_age_increases_over_time(self) -> None:
        decay, clock_state = self._make_decay(start_ts=1_000_000.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.9)
        clock_state[0] += 3600  # advance 1 hour
        age = decay.age_hours("tool_x")
        assert age is not None
        assert abs(age - 1.0) < 0.001

    def test_fresh_signal_no_decay(self) -> None:
        decay, clock_state = self._make_decay(start_ts=1_000_000.0)
        decay.record_signal("gpt-4o", domain="llm", reliability=0.92)
        result = decay.decayed_reliability("gpt-4o", base_reliability=0.92)
        assert result == pytest.approx(0.92, abs=1e-4)

    def test_decay_after_1_hour(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        lam = VolatilityTable.default().lambda_for("llm")
        decay.record_signal("gpt-4o", domain="llm", reliability=0.92)
        clock_state[0] = 3600  # 1 hour later
        result = decay.decayed_reliability("gpt-4o", base_reliability=0.92)
        expected = 0.92 * math.exp(-lam * 1.0)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_decay_after_24_hours(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        lam = VolatilityTable.default().lambda_for("llm")
        decay.record_signal("gpt-4o", domain="llm", reliability=1.0)
        clock_state[0] = 86400  # 24 hours later
        result = decay.decayed_reliability("gpt-4o", base_reliability=1.0)
        expected = math.exp(-lam * 24.0)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_math_domain_decays_slower_than_llm(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("wolfram", domain="math", reliability=1.0)
        decay.record_signal("gpt-4o", domain="llm", reliability=1.0)
        clock_state[0] = 86400  # 24 hours
        r_math = decay.decayed_reliability("wolfram", base_reliability=1.0)
        r_llm = decay.decayed_reliability("gpt-4o", base_reliability=1.0)
        assert r_math > r_llm

    def test_decayed_reliability_clamped_to_1(self) -> None:
        decay, _ = self._make_decay()
        result = decay.decayed_reliability("tool_x", base_reliability=1.5)
        assert result <= 1.0

    def test_decayed_reliability_non_negative(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.01)
        clock_state[0] = 360_000  # very long time
        result = decay.decayed_reliability("tool_x", base_reliability=0.01)
        assert result >= 0.0

    def test_domain_override_in_decayed_reliability(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_x", domain="llm", reliability=1.0)
        clock_state[0] = 3600
        r_math = decay.decayed_reliability("tool_x", base_reliability=1.0, domain="math")
        r_llm = decay.decayed_reliability("tool_x", base_reliability=1.0, domain="llm")
        # math decays slower → higher value
        assert r_math > r_llm

    def test_multiple_tools_independent(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_a", domain="llm", reliability=0.9)
        decay.record_signal("tool_b", domain="math", reliability=0.95)
        clock_state[0] = 7200  # 2 hours
        ra = decay.decayed_reliability("tool_a", base_reliability=0.9)
        rb = decay.decayed_reliability("tool_b", base_reliability=0.95)
        assert ra != rb


class TestReputationDecayEntropy:
    def _make_decay(self, start_ts: float = 0.0) -> tuple[ReputationDecay, list[float]]:
        clock_state = [start_ts]
        decay = ReputationDecay(clock=lambda: clock_state[0])
        return decay, clock_state

    def test_entropy_zero_when_no_signal(self) -> None:
        decay, _ = self._make_decay()
        assert decay.entropy_score("unknown_tool") == 0.0

    def test_entropy_zero_immediately_after_signal(self) -> None:
        decay, _ = self._make_decay(start_ts=1_000_000.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.9)
        assert decay.entropy_score("tool_x") == pytest.approx(0.0, abs=1e-6)

    def test_entropy_increases_with_age(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.9)
        scores = []
        for hours in [1, 6, 24, 72]:
            clock_state[0] = hours * 3600
            scores.append(decay.entropy_score("tool_x"))
        # Entropy is monotonically increasing
        assert scores[0] < scores[1] < scores[2] < scores[3]

    def test_entropy_bounded_0_1(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.5)
        for hours in [0, 1, 100, 10000]:
            clock_state[0] = hours * 3600
            score = decay.entropy_score("tool_x")
            assert 0.0 <= score <= 1.0

    def test_system_entropy_empty_returns_zero(self) -> None:
        decay, _ = self._make_decay()
        assert decay.system_entropy() == 0.0

    def test_system_entropy_mean_of_tools(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_a", domain="llm", reliability=0.9)
        decay.record_signal("tool_b", domain="math", reliability=0.95)
        clock_state[0] = 3600
        sa = decay.entropy_score("tool_a")
        sb = decay.entropy_score("tool_b")
        expected_mean = (sa + sb) / 2
        assert decay.system_entropy() == pytest.approx(expected_mean, abs=1e-9)

    def test_system_entropy_filtered_by_name(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_a", domain="llm", reliability=0.9)
        decay.record_signal("tool_b", domain="math", reliability=0.95)
        clock_state[0] = 3600
        only_a = decay.system_entropy(["tool_a"])
        assert only_a == decay.entropy_score("tool_a")

    def test_all_tool_entropy_returns_dict(self) -> None:
        decay, _ = self._make_decay()
        decay.record_signal("t1", domain="llm", reliability=0.8)
        decay.record_signal("t2", domain="storage", reliability=0.95)
        result = decay.all_tool_entropy()
        assert "t1" in result
        assert "t2" in result

    def test_tracked_tools_sorted(self) -> None:
        decay, _ = self._make_decay()
        decay.record_signal("z_tool", domain="llm", reliability=0.8)
        decay.record_signal("a_tool", domain="llm", reliability=0.8)
        tools = decay.tracked_tools()
        assert tools == sorted(tools)

    def test_re_record_signal_resets_age(self) -> None:
        decay, clock_state = self._make_decay(start_ts=0.0)
        decay.record_signal("tool_x", domain="llm", reliability=0.9)
        clock_state[0] = 7200  # 2 hours; entropy builds up
        e1 = decay.entropy_score("tool_x")
        # New signal resets the clock
        decay.record_signal("tool_x", domain="llm", reliability=0.95)
        e2 = decay.entropy_score("tool_x")
        assert e2 < e1


class TestReputationDecayIntegration:
    """Integration: wire ReputationDecay into ReputationHub."""

    def test_hub_live_reliability_without_gossip_unchanged(self) -> None:
        from manifold.hub import ReputationHub
        hub = ReputationHub()
        # Without any contributions, live_reliability returns baseline (no decay applied)
        r = hub.live_reliability("gpt-4o")
        assert r is not None
        assert 0.0 <= r <= 1.0

    def test_hub_live_reliability_not_none_for_known_tool(self) -> None:
        from manifold.hub import ReputationHub
        hub = ReputationHub()
        r = hub.live_reliability("calculator")
        assert r is not None

    def test_hub_system_entropy_zero_initially(self) -> None:
        from manifold.hub import ReputationHub
        hub = ReputationHub()
        # No gossip contributions yet → no decay signals recorded → entropy = 0
        assert hub.system_entropy() == 0.0

    def test_hub_system_entropy_after_contribution(self) -> None:
        from manifold.hub import ReputationHub
        from manifold.federation import FederatedGossipPacket
        hub = ReputationHub()
        packet = FederatedGossipPacket(
            tool_name="gpt-4o",
            signal="healthy",
            confidence=0.9,
            org_id="test_org",
        )
        hub.contribute(packet)
        # After a contribution the entropy is still ~0 (signal just arrived)
        assert 0.0 <= hub.system_entropy() <= 1.0

    def test_hub_tool_entropy_returns_float(self) -> None:
        from manifold.hub import ReputationHub
        hub = ReputationHub()
        e = hub.tool_entropy("gpt-4o")
        assert isinstance(e, float)
        assert 0.0 <= e <= 1.0
