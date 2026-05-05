"""Tests for Phase 35: Privacy Guard & Anonymisation (privacy.py)."""

from __future__ import annotations

import math

import pytest

from manifold.federation import FederatedGossipPacket
from manifold.privacy import PrivacyConfig, PrivacyGuard
from manifold.threat_feed import ThreatIntelPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_packet(tool: str, signal: str = "failing", org: str = "org-a") -> FederatedGossipPacket:
    return FederatedGossipPacket(
        tool_name=tool, signal=signal, confidence=0.9, org_id=org, weight=1.0  # type: ignore[arg-type]
    )


def _make_threat(
    tool: str,
    event_type: str = "gossip_failing",
    severity: str = "high",
    task_id: str = "task-123",
    org_id: str = "org-x",
) -> ThreatIntelPayload:
    return ThreatIntelPayload(
        event_type=event_type,
        tool_name=tool,
        severity=severity,
        timestamp=1000.0,
        details={"task_id": task_id, "org_id": org_id, "score": 0.5},
        source="manifold",
    )


# ---------------------------------------------------------------------------
# PrivacyGuard construction
# ---------------------------------------------------------------------------


class TestPrivacyGuardInit:
    def test_defaults(self) -> None:
        guard = PrivacyGuard()
        assert guard.k == 5
        assert guard.epsilon == 1.0
        assert guard.sensitivity == 1.0

    def test_custom(self) -> None:
        guard = PrivacyGuard(k=3, epsilon=0.5, sensitivity=2.0)
        assert guard.k == 3
        assert guard.epsilon == 0.5
        assert guard.sensitivity == 2.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            PrivacyGuard(k=0)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PrivacyGuard(epsilon=0.0)

    def test_invalid_sensitivity(self) -> None:
        with pytest.raises(ValueError, match="sensitivity must be > 0"):
            PrivacyGuard(sensitivity=-1.0)

    def test_seed_determinism(self) -> None:
        g1 = PrivacyGuard(seed=42)
        g2 = PrivacyGuard(seed=42)
        v1 = g1.perturb_vector([0.5, 0.5, 0.5])
        v2 = g2.perturb_vector([0.5, 0.5, 0.5])
        assert v1 == v2

    def test_different_seeds_differ(self) -> None:
        g1 = PrivacyGuard(seed=1)
        g2 = PrivacyGuard(seed=2)
        v1 = g1.perturb_vector([0.5] * 10)
        v2 = g2.perturb_vector([0.5] * 10)
        assert v1 != v2


# ---------------------------------------------------------------------------
# k-Anonymity: filter_gossip
# ---------------------------------------------------------------------------


class TestFilterGossip:
    def test_empty_input(self) -> None:
        guard = PrivacyGuard(k=3)
        assert guard.filter_gossip([]) == []

    def test_group_below_k_suppressed(self) -> None:
        guard = PrivacyGuard(k=3)
        packets = [_make_packet("tool-a"), _make_packet("tool-a")]
        result = guard.filter_gossip(packets)
        assert result == []

    def test_group_at_k_passes(self) -> None:
        guard = PrivacyGuard(k=3)
        packets = [_make_packet("tool-a") for _ in range(3)]
        result = guard.filter_gossip(packets)
        assert len(result) == 3

    def test_group_above_k_passes(self) -> None:
        guard = PrivacyGuard(k=2)
        packets = [_make_packet("tool-a") for _ in range(5)]
        result = guard.filter_gossip(packets)
        assert len(result) == 5

    def test_org_id_stripped(self) -> None:
        guard = PrivacyGuard(k=2)
        packets = [_make_packet("tool-a", org="secret-org") for _ in range(3)]
        result = guard.filter_gossip(packets)
        for p in result:
            assert p.org_id == "anonymous"

    def test_tool_name_preserved(self) -> None:
        guard = PrivacyGuard(k=2)
        packets = [_make_packet("tool-b") for _ in range(2)]
        result = guard.filter_gossip(packets)
        assert all(p.tool_name == "tool-b" for p in result)

    def test_mixed_groups(self) -> None:
        guard = PrivacyGuard(k=3)
        packets = (
            [_make_packet("tool-pass") for _ in range(4)]
            + [_make_packet("tool-fail") for _ in range(2)]
        )
        result = guard.filter_gossip(packets)
        tool_names = {p.tool_name for p in result}
        assert "tool-pass" in tool_names
        assert "tool-fail" not in tool_names

    def test_k_equals_1_passes_all(self) -> None:
        guard = PrivacyGuard(k=1)
        packets = [_make_packet("tool-x", org="org-z")]
        result = guard.filter_gossip(packets)
        assert len(result) == 1
        assert result[0].org_id == "anonymous"

    def test_signal_preserved(self) -> None:
        guard = PrivacyGuard(k=2)
        packets = [
            _make_packet("tool-a", signal="healthy"),
            _make_packet("tool-a", signal="failing"),
        ]
        result = guard.filter_gossip(packets)
        signals = {p.signal for p in result}
        assert "healthy" in signals
        assert "failing" in signals

    def test_confidence_preserved(self) -> None:
        guard = PrivacyGuard(k=2)
        packets = [
            FederatedGossipPacket("tool-a", "failing", confidence=0.77, org_id="o1"),
            FederatedGossipPacket("tool-a", "failing", confidence=0.88, org_id="o2"),
        ]
        result = guard.filter_gossip(packets)
        confidences = {round(p.confidence, 2) for p in result}
        assert 0.77 in confidences
        assert 0.88 in confidences


# ---------------------------------------------------------------------------
# k-Anonymity: filter_threat
# ---------------------------------------------------------------------------


class TestFilterThreat:
    def test_empty_input(self) -> None:
        guard = PrivacyGuard(k=3)
        assert guard.filter_threat([]) == []

    def test_group_below_k_suppressed(self) -> None:
        guard = PrivacyGuard(k=3)
        payloads = [_make_threat("tool-a") for _ in range(2)]
        result = guard.filter_threat(payloads)
        assert result == []

    def test_group_at_k_passes(self) -> None:
        guard = PrivacyGuard(k=3)
        payloads = [_make_threat("tool-a") for _ in range(3)]
        result = guard.filter_threat(payloads)
        assert len(result) == 3

    def test_task_id_stripped(self) -> None:
        guard = PrivacyGuard(k=2)
        payloads = [_make_threat("tool-a", task_id="secret-task") for _ in range(2)]
        result = guard.filter_threat(payloads)
        for p in result:
            assert "task_id" not in p.details

    def test_org_id_stripped_from_details(self) -> None:
        guard = PrivacyGuard(k=2)
        payloads = [_make_threat("tool-a", org_id="secret-org") for _ in range(2)]
        result = guard.filter_threat(payloads)
        for p in result:
            assert "org_id" not in p.details

    def test_non_sensitive_details_preserved(self) -> None:
        guard = PrivacyGuard(k=2)
        payloads = [_make_threat("tool-a") for _ in range(2)]
        result = guard.filter_threat(payloads)
        for p in result:
            assert "score" in p.details

    def test_source_normalised(self) -> None:
        guard = PrivacyGuard(k=2)
        payloads = [
            ThreatIntelPayload(
                event_type="gossip_failing",
                tool_name="tool-x",
                severity="high",
                timestamp=1.0,
                details={},
                source="custom-source",
            )
            for _ in range(3)
        ]
        result = guard.filter_threat(payloads)
        assert all(p.source == "manifold" for p in result)

    def test_grouped_by_tool_and_event_type(self) -> None:
        guard = PrivacyGuard(k=3)
        # Two events for (tool-a, gossip_failing) — below k=3, suppressed
        # Three events for (tool-b, canary_fail) — at k=3, passes
        payloads = (
            [_make_threat("tool-a", event_type="gossip_failing") for _ in range(2)]
            + [_make_threat("tool-b", event_type="canary_fail") for _ in range(3)]
        )
        result = guard.filter_threat(payloads)
        tool_names = {p.tool_name for p in result}
        assert "tool-b" in tool_names
        assert "tool-a" not in tool_names


# ---------------------------------------------------------------------------
# Differential privacy: perturb_vector
# ---------------------------------------------------------------------------


class TestPerturbVector:
    def test_empty_vector(self) -> None:
        guard = PrivacyGuard(epsilon=1.0)
        assert guard.perturb_vector([]) == []

    def test_output_length_matches(self) -> None:
        guard = PrivacyGuard(epsilon=1.0)
        v = [0.1, 0.2, 0.3, 0.4]
        result = guard.perturb_vector(v)
        assert len(result) == 4

    def test_noise_is_added(self) -> None:
        guard = PrivacyGuard(seed=99, epsilon=1.0)
        v = [0.5] * 5
        result = guard.perturb_vector(v)
        assert result != v

    def test_deterministic_with_seed(self) -> None:
        g1 = PrivacyGuard(seed=7, epsilon=0.5)
        g2 = PrivacyGuard(seed=7, epsilon=0.5)
        v = [0.3, 0.6, 0.9]
        assert g1.perturb_vector(v) == g2.perturb_vector(v)

    def test_larger_epsilon_means_less_noise(self) -> None:
        """Higher epsilon → smaller scale b = s/ε → smaller expected |noise|."""
        n_samples = 1000
        g_low = PrivacyGuard(seed=1, epsilon=0.1, sensitivity=1.0)
        g_high = PrivacyGuard(seed=1, epsilon=10.0, sensitivity=1.0)
        base = [0.5] * n_samples
        noise_low = [abs(x - 0.5) for x in g_low.perturb_vector(base)]
        noise_high = [abs(x - 0.5) for x in g_high.perturb_vector(base)]
        assert sum(noise_low) > sum(noise_high)

    def test_sensitivity_override(self) -> None:
        g1 = PrivacyGuard(seed=42, epsilon=1.0, sensitivity=1.0)
        g2 = PrivacyGuard(seed=42, epsilon=1.0, sensitivity=1.0)
        v = [0.5]
        r_default = g1.perturb_vector(v)
        r_override = g2.perturb_vector(v, sensitivity=0.5)
        # Different sensitivity → different noise scale → different result
        assert r_default != r_override

    def test_zero_mean_approx(self) -> None:
        """Mean noise over many samples should be close to 0 (Laplace is zero-mean)."""
        guard = PrivacyGuard(seed=0, epsilon=2.0)
        base = [0.0] * 10_000
        result = guard.perturb_vector(base)
        mean_noise = sum(result) / len(result)
        assert abs(mean_noise) < 0.1  # within 10% of zero

    def test_laplace_scale_sanity(self) -> None:
        """b = sensitivity/epsilon; about 50% of |samples| should be ≤ b."""
        b = 1.0  # epsilon=1, sensitivity=1
        guard = PrivacyGuard(seed=0, epsilon=1.0, sensitivity=1.0)
        base = [0.0] * 5_000
        noises = [abs(x) for x in guard.perturb_vector(base)]
        # Laplace CDF: P(|X| ≤ b) = 1 - exp(-1) ≈ 0.632
        frac_within = sum(1 for n in noises if n <= b) / len(noises)
        assert 0.5 < frac_within < 0.8


# ---------------------------------------------------------------------------
# Summary and config
# ---------------------------------------------------------------------------


class TestSummaryAndConfig:
    def test_summary_keys(self) -> None:
        guard = PrivacyGuard(k=4, epsilon=2.0, sensitivity=0.5)
        s = guard.summary()
        assert s["k"] == 4
        assert s["epsilon"] == 2.0
        assert s["sensitivity"] == 0.5
        assert math.isclose(s["laplace_scale"], 0.25)  # type: ignore[arg-type]

    def test_config_snapshot(self) -> None:
        guard = PrivacyGuard(k=7, epsilon=0.3, seed=99)
        cfg = guard.config()
        assert isinstance(cfg, PrivacyConfig)
        assert cfg.k == 7
        assert cfg.epsilon == 0.3
        assert cfg.seed == 99

    def test_config_is_frozen(self) -> None:
        guard = PrivacyGuard()
        cfg = guard.config()
        with pytest.raises(Exception):
            cfg.k = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_perturb(self) -> None:
        import threading

        guard = PrivacyGuard(epsilon=1.0)
        results: list[list[float]] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                results.append(guard.perturb_vector([0.5] * 50))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        # All results should have the correct length
        assert all(len(r) == 50 for r in results)
