"""Tests for Phase 39: Autonomic PID Controller (manifold/pid.py)."""

from __future__ import annotations

import pytest

from manifold.interceptor import InterceptorConfig
from manifold.pid import PIDConfig, PIDState, RiskPIDController


# ---------------------------------------------------------------------------
# PIDConfig
# ---------------------------------------------------------------------------


class TestPIDConfig:
    def test_defaults(self) -> None:
        cfg = PIDConfig()
        assert cfg.kp == 1.0
        assert cfg.ki == 0.1
        assert cfg.kd == 0.05
        assert cfg.setpoint == 0.3
        assert cfg.output_min == 0.1
        assert cfg.output_max == 0.9
        assert cfg.integral_limit == 5.0

    def test_custom(self) -> None:
        cfg = PIDConfig(kp=2.0, ki=0.5, kd=0.1, setpoint=0.5)
        assert cfg.kp == 2.0
        assert cfg.setpoint == 0.5

    def test_frozen(self) -> None:
        cfg = PIDConfig()
        with pytest.raises(Exception):
            cfg.kp = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PIDState
# ---------------------------------------------------------------------------


class TestPIDState:
    def _make_state(self) -> PIDState:
        return PIDState(
            timestamp=1000.0,
            measured_value=0.5,
            error=-0.2,
            proportional=-0.2,
            integral=-0.02,
            derivative=0.0,
            output=0.3,
            threshold_before=0.45,
            threshold_after=0.3,
        )

    def test_creation(self) -> None:
        s = self._make_state()
        assert s.measured_value == 0.5
        assert s.error == pytest.approx(-0.2)

    def test_to_dict(self) -> None:
        s = self._make_state()
        d = s.to_dict()
        assert d["measured_value"] == pytest.approx(0.5)
        assert d["output"] == pytest.approx(0.3)
        assert "threshold_after" in d

    def test_frozen(self) -> None:
        s = self._make_state()
        with pytest.raises(Exception):
            s.output = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RiskPIDController
# ---------------------------------------------------------------------------


class TestRiskPIDControllerBasics:
    def test_tick_returns_state(self) -> None:
        pid = RiskPIDController()
        state = pid.tick(measured_value=0.5)
        assert isinstance(state, PIDState)

    def test_history_accumulates(self) -> None:
        pid = RiskPIDController()
        pid.tick(measured_value=0.3)
        pid.tick(measured_value=0.4)
        pid.tick(measured_value=0.5)
        assert len(pid.history()) == 3

    def test_last_state(self) -> None:
        pid = RiskPIDController()
        assert pid.last_state() is None
        pid.tick(measured_value=0.3)
        assert pid.last_state() is not None

    def test_reset_clears_history(self) -> None:
        pid = RiskPIDController()
        pid.tick(0.3)
        pid.tick(0.5)
        pid.reset()
        assert len(pid.history()) == 0
        assert pid.last_state() is None

    def test_summary_keys(self) -> None:
        pid = RiskPIDController()
        pid.tick(0.3)
        s = pid.summary()
        for k in ("total_ticks", "integral", "last_output", "last_error", "setpoint", "kp", "ki", "kd"):
            assert k in s

    def test_summary_total_ticks(self) -> None:
        pid = RiskPIDController()
        pid.tick(0.2)
        pid.tick(0.3)
        assert pid.summary()["total_ticks"] == 2


class TestRiskPIDControllerMath:
    def test_output_clamped_min(self) -> None:
        """When entropy >> setpoint, negative error → low output → clamped to output_min."""
        cfg = PIDConfig(kp=10.0, ki=0.0, kd=0.0, setpoint=0.1, output_min=0.1)
        pid = RiskPIDController(config=cfg)
        state = pid.tick(measured_value=0.9)
        # e = 0.1 - 0.9 = -0.8, u = 10 * -0.8 = -8 → clamped to 0.1
        assert state.output == pytest.approx(0.1)

    def test_output_clamped_max(self) -> None:
        """When entropy << setpoint, large positive error → output_max."""
        cfg = PIDConfig(kp=10.0, ki=0.0, kd=0.0, setpoint=0.9, output_max=0.9)
        pid = RiskPIDController(config=cfg)
        state = pid.tick(measured_value=0.1)
        # e = 0.9 - 0.1 = 0.8, u = 10 * 0.8 = 8 → clamped to 0.9
        assert state.output == pytest.approx(0.9)

    def test_zero_error_proportional_zero(self) -> None:
        cfg = PIDConfig(kp=2.0, ki=0.0, kd=0.0, setpoint=0.5)
        pid = RiskPIDController(config=cfg)
        state = pid.tick(measured_value=0.5)
        # e = 0, P = 0, I = 0 (first tick), D = 0
        assert state.proportional == pytest.approx(0.0)

    def test_proportional_term(self) -> None:
        cfg = PIDConfig(kp=2.0, ki=0.0, kd=0.0, setpoint=0.5)
        pid = RiskPIDController(config=cfg)
        state = pid.tick(measured_value=0.3)
        # e = 0.5 - 0.3 = 0.2; P = 2.0 * 0.2 = 0.4
        assert state.proportional == pytest.approx(0.4)

    def test_integral_accumulates(self) -> None:
        cfg = PIDConfig(kp=0.0, ki=1.0, kd=0.0, setpoint=0.5, output_min=0.0, output_max=10.0)
        pid = RiskPIDController(config=cfg)
        # First tick: e = 0.5 - 0.3 = 0.2; integral = 0.2 * 1.0 = 0.2
        s1 = pid.tick(measured_value=0.3)
        assert s1.integral > 0.0
        # Second tick adds more
        s2 = pid.tick(measured_value=0.3)
        assert s2.integral > s1.integral

    def test_anti_windup_clamp(self) -> None:
        """Integral should not exceed integral_limit."""
        cfg = PIDConfig(kp=0.0, ki=1.0, kd=0.0, setpoint=1.0, integral_limit=0.5, output_min=0.0, output_max=100.0)
        pid = RiskPIDController(config=cfg)
        for _ in range(100):
            pid.tick(measured_value=0.0)
        # integral should be clamped at 0.5
        assert abs(pid._integral) <= 0.5 + 1e-9  # noqa: SLF001

    def test_derivative_zero_first_tick(self) -> None:
        cfg = PIDConfig(kp=0.0, ki=0.0, kd=5.0, setpoint=0.5)
        pid = RiskPIDController(config=cfg)
        state = pid.tick(measured_value=0.3)
        assert state.derivative == pytest.approx(0.0)

    def test_derivative_nonzero_second_tick(self) -> None:
        cfg = PIDConfig(kp=0.0, ki=0.0, kd=1.0, setpoint=0.5)
        pid = RiskPIDController(config=cfg)
        pid.tick(measured_value=0.3)
        state2 = pid.tick(measured_value=0.4)
        # Error changed: e1=0.2, e2=0.1, de=-0.1 → derivative should be non-zero
        assert state2.derivative != 0.0

    def test_error_stored_in_state(self) -> None:
        pid = RiskPIDController(config=PIDConfig(setpoint=0.6))
        state = pid.tick(measured_value=0.4)
        assert state.error == pytest.approx(0.2)


class TestRiskPIDControllerInterceptorIntegration:
    def test_updates_interceptor_config(self) -> None:
        cfg = InterceptorConfig(risk_veto_threshold=0.45)
        pid = RiskPIDController(
            config=PIDConfig(kp=5.0, ki=0.0, kd=0.0, setpoint=0.5),
            interceptor_config=cfg,
        )
        state = pid.tick(measured_value=0.1)
        # The threshold should now equal the PID output
        assert cfg.risk_veto_threshold == pytest.approx(state.threshold_after)
        assert cfg.risk_veto_threshold != pytest.approx(0.45)

    def test_threshold_before_after_differ_when_changed(self) -> None:
        cfg = InterceptorConfig(risk_veto_threshold=0.45)
        pid = RiskPIDController(
            config=PIDConfig(kp=5.0, ki=0.0, kd=0.0, setpoint=0.5),
            interceptor_config=cfg,
        )
        state = pid.tick(measured_value=0.1)
        assert state.threshold_before == pytest.approx(0.45)
        assert state.threshold_after != pytest.approx(0.45)

    def test_read_only_mode_no_interceptor(self) -> None:
        pid = RiskPIDController()
        state = pid.tick(measured_value=0.3)
        # No exception; threshold_before defaults to 0.45
        assert state.threshold_before == pytest.approx(0.45)

    def test_entropy_source_called(self) -> None:
        calls: list[float] = []
        def entropy_fn() -> float:
            calls.append(1.0)
            return 0.4

        pid = RiskPIDController(entropy_source=entropy_fn)
        pid.tick()  # no explicit measured_value
        assert len(calls) == 1

    def test_entropy_source_used_in_output(self) -> None:
        pid = RiskPIDController(entropy_source=lambda: 0.3)
        state = pid.tick()
        # With setpoint=0.3 and measured=0.3, error=0
        assert state.error == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestRiskPIDEdgeCases:
    def test_multiple_resets(self) -> None:
        pid = RiskPIDController()
        for _ in range(5):
            pid.tick(0.5)
        pid.reset()
        pid.reset()  # double reset is safe
        assert len(pid.history()) == 0

    def test_history_immutable_copy(self) -> None:
        pid = RiskPIDController()
        pid.tick(0.3)
        h1 = pid.history()
        pid.tick(0.4)
        h2 = pid.history()
        assert len(h1) == 1
        assert len(h2) == 2

    def test_output_within_bounds(self) -> None:
        cfg = PIDConfig(kp=100.0, ki=100.0, kd=100.0)
        pid = RiskPIDController(config=cfg)
        for v in [0.0, 0.5, 1.0]:
            state = pid.tick(measured_value=v)
            assert cfg.output_min <= state.output <= cfg.output_max

    def test_summary_after_reset(self) -> None:
        pid = RiskPIDController()
        pid.tick(0.3)
        pid.reset()
        s = pid.summary()
        assert s["total_ticks"] == 0
        assert s["last_output"] == 0.0

    def test_kd_zero_pure_pi(self) -> None:
        cfg = PIDConfig(kp=1.0, ki=0.5, kd=0.0, setpoint=0.4)
        pid = RiskPIDController(config=cfg)
        s1 = pid.tick(measured_value=0.4)
        assert s1.derivative == pytest.approx(0.0)

