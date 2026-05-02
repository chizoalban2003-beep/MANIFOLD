"""Tests for Phase 8: ToolConnector, ConnectorRegistry, ShadowModeWrapper, VirtualRegret."""

import pytest

from manifold import (
    BrainConfig,
    BrainDecision,
    BrainTask,
    ConnectorRegistry,
    ManifoldBrain,
    ShadowModeWrapper,
    ToolConnector,
    ToolConnectorResult,
    ToolProfile,
    VirtualRegret,
    default_tools,
)
from manifold.connector import _classify_error


_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)
_TOOLS = default_tools()


# ---------------------------------------------------------------------------
# _classify_error helper
# ---------------------------------------------------------------------------


def test_classify_timeout() -> None:
    assert _classify_error("TimeoutError") == "timeout"


def test_classify_http_error() -> None:
    assert _classify_error("HTTPError") == "tool_error"


def test_classify_value_error() -> None:
    assert _classify_error("ValueError") == "bad_data"


def test_classify_unknown() -> None:
    assert _classify_error("SomethingWeird") == "unknown"


def test_classify_rate_limit() -> None:
    assert _classify_error("RateLimitError") == "rate_limit"


# ---------------------------------------------------------------------------
# ToolConnectorResult.to_brain_outcome
# ---------------------------------------------------------------------------


def test_connector_result_success_outcome() -> None:
    result = ToolConnectorResult(
        tool_name="test", output="ok", success=True, latency_seconds=0.05
    )
    outcome = result.to_brain_outcome()
    assert outcome.success
    assert outcome.asset_gained > 0.0
    assert outcome.risk_realized == 0.0


def test_connector_result_failure_outcome() -> None:
    result = ToolConnectorResult(
        tool_name="test", output=None, success=False,
        latency_seconds=0.20, error_type="TimeoutError"
    )
    outcome = result.to_brain_outcome()
    assert not outcome.success
    assert outcome.failure_mode == "timeout"
    assert outcome.risk_realized > 0.0


def test_connector_result_cost_scales_with_latency() -> None:
    fast = ToolConnectorResult("t", "ok", True, 0.10).to_brain_outcome(cost_per_second=0.10)
    slow = ToolConnectorResult("t", "ok", True, 1.00).to_brain_outcome(cost_per_second=0.10)
    assert slow.cost_paid > fast.cost_paid


# ---------------------------------------------------------------------------
# ToolConnector — construction
# ---------------------------------------------------------------------------


def test_tool_connector_default_profile() -> None:
    tc = ToolConnector(name="my_tool", fn=lambda: "result")
    assert tc.profile.name == "my_tool"
    assert 0.0 < tc.profile.reliability <= 1.0


def test_tool_connector_custom_profile() -> None:
    profile = ToolProfile("custom", cost=0.05, latency=0.05, reliability=0.99, risk=0.01, asset=0.90)
    tc = ToolConnector(name="custom", fn=lambda: 42, profile=profile)
    assert tc.profile.reliability == 0.99


# ---------------------------------------------------------------------------
# ToolConnector — call / telemetry
# ---------------------------------------------------------------------------


def test_tool_connector_call_success() -> None:
    tc = ToolConnector(name="add", fn=lambda x, y: x + y)
    result = tc.call(2, 3)
    assert result.success
    assert result.output == 5
    assert result.latency_seconds >= 0.0


def test_tool_connector_call_failure_caught() -> None:
    def bad_fn() -> None:
        raise ValueError("invalid data")

    tc = ToolConnector(name="broken", fn=bad_fn)
    result = tc.call()
    assert not result.success
    assert result.error_type == "ValueError"
    assert "invalid data" in result.error_message


def test_tool_connector_call_count() -> None:
    tc = ToolConnector(name="counter", fn=lambda: 1)
    tc.call()
    tc.call()
    assert tc.call_count() == 2


def test_tool_connector_failure_count() -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    tc = ToolConnector(name="fail", fn=fail)
    tc.call()
    tc.call()
    assert tc.failure_count() == 2


def test_tool_connector_observed_reliability_all_success() -> None:
    tc = ToolConnector(name="reliable", fn=lambda: "ok")
    for _ in range(10):
        tc.call()
    assert tc.observed_reliability() == pytest.approx(1.0)


def test_tool_connector_observed_reliability_mixed() -> None:
    calls = [True, True, False, True, False]

    def mixed(i: int) -> str:
        if not calls[i]:
            raise RuntimeError("fail")
        return "ok"

    tc = ToolConnector(name="mixed", fn=mixed)
    for i in range(len(calls)):
        tc.call(i)
    obs = tc.observed_reliability()
    assert 0.0 <= obs <= 1.0


def test_tool_connector_no_calls_reliability_one() -> None:
    tc = ToolConnector(name="unused", fn=lambda: None)
    assert tc.observed_reliability() == 1.0


def test_tool_connector_mean_latency_positive() -> None:
    import time
    tc = ToolConnector(name="slow", fn=lambda: time.sleep(0.001) or "ok")
    tc.call()
    assert tc.mean_latency() > 0.0


def test_tool_connector_refreshed_profile_blends() -> None:
    profile = ToolProfile("x", cost=0.10, latency=0.15, reliability=0.90, risk=0.05, asset=0.80)
    tc = ToolConnector(name="x", fn=lambda: "ok", profile=profile)
    # 20 successful calls → full telemetry weight
    for _ in range(20):
        tc.call()
    refreshed = tc.refreshed_profile()
    assert refreshed.reliability == pytest.approx(1.0, abs=0.05)  # all successes → near 1.0


# ---------------------------------------------------------------------------
# ConnectorRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    reg = ConnectorRegistry()
    tc = ToolConnector(name="search", fn=lambda q: f"results for {q}")
    reg.register(tc)
    assert reg.get("search") is tc


def test_registry_get_missing_returns_none() -> None:
    reg = ConnectorRegistry()
    assert reg.get("missing") is None


def test_registry_names() -> None:
    reg = ConnectorRegistry()
    reg.register(ToolConnector("a", lambda: None))
    reg.register(ToolConnector("b", lambda: None))
    assert set(reg.names()) == {"a", "b"}


def test_registry_len() -> None:
    reg = ConnectorRegistry()
    reg.register(ToolConnector("x", lambda: None))
    assert len(reg) == 1


def test_registry_tool_profiles_with_telemetry() -> None:
    reg = ConnectorRegistry()
    tc = ToolConnector("calc", lambda x: x * 2)
    for _ in range(5):
        tc.call(3)
    reg.register(tc)
    profiles = reg.tool_profiles(use_telemetry=True)
    assert len(profiles) == 1
    assert profiles[0].name == "calc"


def test_registry_tool_profiles_without_telemetry() -> None:
    reg = ConnectorRegistry()
    original_profile = ToolProfile("t", cost=0.10, latency=0.20, reliability=0.70, risk=0.10, asset=0.60)
    tc = ToolConnector("t", lambda: None, profile=original_profile)
    reg.register(tc)
    profiles = reg.tool_profiles(use_telemetry=False)
    assert profiles[0].reliability == pytest.approx(0.70)


def test_registry_overwrite_existing() -> None:
    reg = ConnectorRegistry()
    reg.register(ToolConnector("dup", lambda: 1))
    reg.register(ToolConnector("dup", lambda: 2))
    assert len(reg) == 1


# ---------------------------------------------------------------------------
# ShadowModeWrapper
# ---------------------------------------------------------------------------


def test_shadow_wrapper_observe_returns_virtual_regret() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("Test task", "general")
    vr = wrapper.observe(task, actual_action="answer")
    assert isinstance(vr, VirtualRegret)


def test_shadow_wrapper_regret_zero_on_agreement() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("Simple task", "general")
    # Get what brain would decide, pass it as actual
    real_decision = brain.decide(task)
    vr = wrapper.observe(task, actual_action=real_decision.action)
    assert vr.regret == 0
    assert not vr.is_disagreement


def test_shadow_wrapper_regret_one_on_disagreement() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("Refactor entire codebase", "coding", complexity=0.9, stakes=0.9)
    vr = wrapper.observe(task, actual_action="definitely_not_a_real_action_xyz")
    # MANIFOLD would never choose "definitely_not_a_real_action_xyz"
    assert vr.regret == 1
    assert vr.is_disagreement


def test_shadow_wrapper_accumulates_log() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("hello", "general")
    for _ in range(5):
        wrapper.observe(task, "answer")
    assert len(wrapper.virtual_regret_log()) == 5


def test_shadow_wrapper_total_regret() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("hello", "general")
    for _ in range(3):
        wrapper.observe(task, "fake_action_never_chosen")
    assert wrapper.total_regret() == 3


def test_shadow_wrapper_disagreement_rate() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    assert wrapper.disagreement_rate() == 0.0
    task = BrainTask("hello", "general")
    wrapper.observe(task, "fake_action")
    assert wrapper.disagreement_rate() > 0.0


def test_shadow_wrapper_shadow_report_structure() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("hello", "general")
    wrapper.observe(task, "answer")
    report = wrapper.shadow_report()
    assert "total_observations" in report
    assert "disagreement_rate" in report
    assert "active" in report
    assert report["total_observations"] == 1


def test_shadow_wrapper_activate_deactivate() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    assert not wrapper.active
    wrapper.activate()
    assert wrapper.active
    wrapper.deactivate()
    assert not wrapper.active


def test_shadow_wrapper_top_disagreements_in_report() -> None:
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    wrapper = ShadowModeWrapper(brain=brain)
    task = BrainTask("hello", "general")
    for _ in range(5):
        wrapper.observe(task, "always_wrong_action")
    report = wrapper.shadow_report()
    assert isinstance(report["top_disagreement_actions"], list)
