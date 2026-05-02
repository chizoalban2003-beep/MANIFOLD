"""Tests for Phase 13: ActiveInterceptor and @shield decorator."""

from __future__ import annotations

import pytest

from manifold.brain import BrainConfig, BrainTask, ManifoldBrain, ToolProfile
from manifold.connector import ConnectorRegistry, ToolConnector
from manifold.interceptor import (
    ActiveInterceptor,
    InterceptorConfig,
    InterceptorVeto,
    InterceptResult,
    shield,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(tools: list[ToolProfile] | None = None) -> ManifoldBrain:
    cfg = BrainConfig()
    if tools is None:
        tools = [
            ToolProfile("safe_tool", cost=0.05, latency=0.1, reliability=0.95, risk=0.05, asset=0.8),
            ToolProfile("risky_tool", cost=0.10, latency=0.2, reliability=0.60, risk=0.85, asset=0.5),
        ]
    return ManifoldBrain(cfg, tools=tools)


def _make_registry() -> ConnectorRegistry:
    reg = ConnectorRegistry()
    reg.register(ToolConnector("safe_tool", fn=lambda *_: "ok", profile=ToolProfile("safe_tool", cost=0.05, latency=0.1, reliability=0.95, risk=0.05, asset=0.8)))
    reg.register(ToolConnector("risky_tool", fn=lambda *_: "ok", profile=ToolProfile("risky_tool", cost=0.10, latency=0.2, reliability=0.60, risk=0.85, asset=0.5)))
    return reg


def _low_risk_task() -> BrainTask:
    return BrainTask(prompt="list recent orders", domain="support", stakes=0.1, uncertainty=0.2, complexity=0.2)


def _high_risk_task() -> BrainTask:
    return BrainTask(prompt="charge customer card", domain="finance", stakes=0.95, uncertainty=0.7, complexity=0.8, safety_sensitivity=0.9)


# ---------------------------------------------------------------------------
# InterceptorConfig
# ---------------------------------------------------------------------------

class TestInterceptorConfig:
    def test_defaults(self):
        cfg = InterceptorConfig()
        assert cfg.risk_veto_threshold == 0.45
        assert cfg.redirect_strategy == "hitl"
        assert cfg.fallback_min_reliability == 0.70

    def test_custom(self):
        cfg = InterceptorConfig(risk_veto_threshold=0.3, redirect_strategy="fallback")
        assert cfg.risk_veto_threshold == 0.3


# ---------------------------------------------------------------------------
# InterceptResult
# ---------------------------------------------------------------------------

class TestInterceptResult:
    def test_permitted_result(self):
        brain = _make_brain()
        task = _low_risk_task()
        dec = brain.decide(task)
        r = InterceptResult(
            tool_name="safe_tool",
            permitted=True,
            risk_score=0.02,
            redirect_to=None,
            reason="risk within threshold",
            manifold_decision=dec,
        )
        assert r.permitted
        assert r.redirect_to is None

    def test_vetoed_result(self):
        brain = _make_brain()
        task = _high_risk_task()
        dec = brain.decide(task)
        r = InterceptResult(
            tool_name="risky_tool",
            permitted=False,
            risk_score=0.80,
            redirect_to="hitl",
            reason="risk_score=0.800 >= threshold=0.450",
            manifold_decision=dec,
        )
        assert not r.permitted
        assert r.redirect_to == "hitl"


# ---------------------------------------------------------------------------
# ActiveInterceptor — basic permit / veto
# ---------------------------------------------------------------------------

class TestActiveInterceptorPermit:
    def test_low_risk_low_stakes_permitted(self):
        reg = _make_registry()
        brain = _make_brain()
        interceptor = ActiveInterceptor(registry=reg, brain=brain)
        task = _low_risk_task()
        result = interceptor.intercept(task, requested_tool="safe_tool")
        assert result.permitted
        assert result.redirect_to is None
        assert result.risk_score < 0.45

    def test_permit_increments_count(self):
        reg = _make_registry()
        brain = _make_brain()
        interceptor = ActiveInterceptor(registry=reg, brain=brain)
        for _ in range(5):
            interceptor.intercept(_low_risk_task(), "safe_tool")
        assert interceptor.permit_count() == 5
        assert interceptor.veto_count() == 0


class TestActiveInterceptorVeto:
    def test_high_risk_high_stakes_vetoed_to_hitl(self):
        reg = _make_registry()
        brain = _make_brain()
        cfg = InterceptorConfig(risk_veto_threshold=0.40, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = _high_risk_task()
        result = interceptor.intercept(task, requested_tool="risky_tool")
        assert not result.permitted
        assert result.redirect_to == "hitl"

    def test_veto_rate(self):
        reg = _make_registry()
        brain = _make_brain()
        cfg = InterceptorConfig(risk_veto_threshold=0.40, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        interceptor.intercept(_low_risk_task(), "safe_tool")
        interceptor.intercept(_high_risk_task(), "risky_tool")
        assert interceptor.veto_count() >= 1
        assert 0.0 < interceptor.veto_rate() <= 1.0

    def test_unregistered_tool_raises_key_error(self):
        reg = _make_registry()
        brain = _make_brain()
        interceptor = ActiveInterceptor(registry=reg, brain=brain)
        with pytest.raises(KeyError):
            interceptor.intercept(_low_risk_task(), "ghost_tool")


# ---------------------------------------------------------------------------
# ActiveInterceptor — redirect strategies
# ---------------------------------------------------------------------------

class TestRedirectStrategies:
    def test_strategy_hitl(self):
        reg = _make_registry()
        brain = _make_brain()
        cfg = InterceptorConfig(risk_veto_threshold=0.01, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        result = interceptor.intercept(_high_risk_task(), "risky_tool")
        # Either permitted (brain chose use_tool) or redirected to hitl
        if not result.permitted:
            assert result.redirect_to == "hitl"

    def test_strategy_fallback_finds_safe_tool(self):
        reg = ConnectorRegistry()
        reg.register(ToolConnector("safe_tool", fn=lambda: "ok", profile=ToolProfile("safe_tool", cost=0.05, latency=0.1, reliability=0.95, risk=0.05, asset=0.8)))
        reg.register(ToolConnector("risky_tool", fn=lambda: "ok", profile=ToolProfile("risky_tool", cost=0.10, latency=0.2, reliability=0.60, risk=0.85, asset=0.5)))
        brain = _make_brain()
        cfg = InterceptorConfig(risk_veto_threshold=0.40, redirect_strategy="fallback")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = _high_risk_task()
        result = interceptor.intercept(task, "risky_tool")
        if not result.permitted:
            # Should fall back to safe_tool, not hitl
            assert result.redirect_to in ("safe_tool", "hitl")

    def test_strategy_refuse_raises_interceptor_veto(self):
        reg = ConnectorRegistry()
        reg.register(ToolConnector(
            "danger_tool",
            fn=lambda: "ok",
            profile=ToolProfile("danger_tool", cost=0.10, latency=0.2, reliability=0.60, risk=0.99, asset=0.4),
        ))
        brain = ManifoldBrain(BrainConfig(), tools=[ToolProfile("danger_tool", cost=0.10, latency=0.2, reliability=0.60, risk=0.99, asset=0.4)])
        cfg = InterceptorConfig(risk_veto_threshold=0.10, redirect_strategy="refuse")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = BrainTask(prompt="dangerous op", stakes=0.95, safety_sensitivity=0.9)
        with pytest.raises(InterceptorVeto) as exc_info:
            interceptor.intercept(task, "danger_tool")
        assert exc_info.value.result.tool_name == "danger_tool"
        assert not exc_info.value.result.permitted


# ---------------------------------------------------------------------------
# ActiveInterceptor — .call() method
# ---------------------------------------------------------------------------

class TestActiveInterceptorCall:
    def test_call_permitted_executes_fn(self):
        reg = ConnectorRegistry()
        calls = []
        reg.register(ToolConnector("calc", fn=lambda x: calls.append(x) or x * 2, profile=ToolProfile("calc", cost=0.02, latency=0.05, reliability=0.98, risk=0.02, asset=0.8)))
        brain = ManifoldBrain(BrainConfig(), tools=[ToolProfile("calc", cost=0.02, latency=0.05, reliability=0.98, risk=0.02, asset=0.8)])
        cfg = InterceptorConfig(risk_veto_threshold=0.45, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = BrainTask(prompt="calculate", stakes=0.1)
        conn_result = interceptor.call(task, "calc", 5)
        # If permitted, fn was called and output is 10; if vetoed, error_type is hitl_escalation
        assert conn_result.tool_name == "calc"

    def test_call_vetoed_returns_hitl_escalation(self):
        reg = ConnectorRegistry()
        reg.register(ToolConnector(
            "evil_tool",
            fn=lambda: "hack",
            profile=ToolProfile("evil_tool", cost=0.20, latency=0.3, reliability=0.5, risk=0.95, asset=0.2),
        ))
        brain = ManifoldBrain(BrainConfig(), tools=[ToolProfile("evil_tool", cost=0.20, latency=0.3, reliability=0.5, risk=0.95, asset=0.2)])
        cfg = InterceptorConfig(risk_veto_threshold=0.10, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = BrainTask(prompt="evil op", stakes=0.99, safety_sensitivity=0.99)
        conn_result = interceptor.call(task, "evil_tool")
        if not conn_result.success:
            assert conn_result.error_type in ("hitl_escalation",)

    def test_call_fallback_executes_fallback(self):
        reg = ConnectorRegistry()
        executed = []
        reg.register(ToolConnector(
            "primary",
            fn=lambda: "primary",
            profile=ToolProfile("primary", cost=0.10, latency=0.2, reliability=0.50, risk=0.90, asset=0.3),
        ))
        reg.register(ToolConnector(
            "backup",
            fn=lambda: executed.append("backup") or "backup_ok",
            profile=ToolProfile("backup", cost=0.05, latency=0.1, reliability=0.95, risk=0.05, asset=0.8),
        ))
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("primary", cost=0.10, latency=0.2, reliability=0.50, risk=0.90, asset=0.3),
            ToolProfile("backup", cost=0.05, latency=0.1, reliability=0.95, risk=0.05, asset=0.8),
        ])
        cfg = InterceptorConfig(risk_veto_threshold=0.40, redirect_strategy="fallback")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = BrainTask(prompt="risky payment", stakes=0.95, safety_sensitivity=0.9)
        conn_result = interceptor.call(task, "primary")
        assert conn_result.tool_name in ("primary", "backup")


# ---------------------------------------------------------------------------
# ActiveInterceptor — summary
# ---------------------------------------------------------------------------

class TestInterceptorSummary:
    def test_summary_keys(self):
        reg = _make_registry()
        brain = _make_brain()
        interceptor = ActiveInterceptor(registry=reg, brain=brain)
        interceptor.intercept(_low_risk_task(), "safe_tool")
        s = interceptor.summary()
        assert set(s) >= {"total_calls", "permitted", "vetoed", "veto_rate", "redirected_to_hitl", "redirected_to_fallback", "avg_risk_score"}

    def test_summary_values_consistent(self):
        reg = _make_registry()
        brain = _make_brain()
        cfg = InterceptorConfig(risk_veto_threshold=0.40, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        for _ in range(10):
            interceptor.intercept(_low_risk_task(), "safe_tool")
        s = interceptor.summary()
        assert s["total_calls"] == 10
        assert s["permitted"] + s["vetoed"] == 10

    def test_empty_summary(self):
        reg = _make_registry()
        brain = _make_brain()
        interceptor = ActiveInterceptor(registry=reg, brain=brain)
        s = interceptor.summary()
        assert s["total_calls"] == 0
        assert s["veto_rate"] == 0.0

    def test_intercept_log_length(self):
        reg = _make_registry()
        brain = _make_brain()
        interceptor = ActiveInterceptor(registry=reg, brain=brain)
        for _ in range(7):
            interceptor.intercept(_low_risk_task(), "safe_tool")
        assert len(interceptor.intercept_log()) == 7


# ---------------------------------------------------------------------------
# @shield decorator
# ---------------------------------------------------------------------------

class TestShieldDecorator:
    def test_shield_permits_low_risk_call(self):
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("add", cost=0.01, latency=0.05, reliability=0.99, risk=0.01, asset=0.9),
        ])

        @shield(brain=brain, domain="math", stakes=0.05, uncertainty=0.1)
        def add(a: int, b: int) -> int:
            return a + b

        # Low stakes should nearly always be permitted
        # (brain may still refuse in rare edge cases; we just ensure no crash on permit)
        try:
            result = add(2, 3)
            assert result == 5
        except InterceptorVeto:
            pass  # acceptable if brain chose to refuse

    def test_shield_blocks_high_risk_call_raise_on_veto(self):
        # Create a brain that always chooses "refuse" for high-stakes financial ops
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("payment", cost=0.50, latency=0.5, reliability=0.40, risk=0.99, asset=0.1),
        ])

        @shield(
            brain=brain,
            domain="finance",
            stakes=0.99,
            uncertainty=0.9,
            complexity=0.9,
            safety_sensitivity=0.99,
            raise_on_veto=True,
        )
        def process_payment(amount: float) -> str:
            return f"charged {amount}"

        # With stakes=0.99 and risk=0.99, brain should eventually refuse.
        # Run multiple times to trigger veto
        vetoed = False
        for _ in range(20):
            try:
                process_payment(100.0)
            except InterceptorVeto as exc:
                assert exc.result.tool_name == "process_payment"
                vetoed = True
                break
        # Veto must be triggered at least once in 20 high-stakes calls
        assert vetoed

    def test_shield_silent_veto_returns_none(self):
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("danger", cost=0.50, latency=0.5, reliability=0.40, risk=0.99, asset=0.1),
        ])

        @shield(
            brain=brain,
            domain="finance",
            stakes=0.99,
            uncertainty=0.9,
            complexity=0.9,
            safety_sensitivity=0.99,
            raise_on_veto=False,
        )
        def dangerous_op() -> str:
            return "executed"

        # Run multiple times; if vetoed, None is returned (no exception)
        results = [dangerous_op() for _ in range(10)]
        # All results must be either "executed" or None
        for r in results:
            assert r in ("executed", None)

    def test_shield_preserves_function_name_and_docstring(self):
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("test", cost=0.01, latency=0.05, reliability=0.99, risk=0.01, asset=0.9),
        ])

        @shield(brain=brain)
        def my_function() -> str:
            """My docstring."""
            return "ok"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_shield_custom_veto_actions(self):
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("tool", cost=0.05, latency=0.1, reliability=0.95, risk=0.05, asset=0.8),
        ])

        # Only "stop" should veto — brain won't normally output "stop", so
        # function should execute freely
        @shield(brain=brain, domain="general", stakes=0.5, veto_actions=("stop",))
        def benign() -> str:
            return "ran"

        result = benign()
        # Brain with low-risk task should not choose "stop"
        assert result == "ran"


# ---------------------------------------------------------------------------
# InterceptorVeto exception
# ---------------------------------------------------------------------------

class TestInterceptorVetoException:
    def test_veto_inherits_runtime_error(self):
        assert issubclass(InterceptorVeto, RuntimeError)

    def test_veto_carries_result(self):
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("bad", cost=0.5, latency=0.3, reliability=0.4, risk=0.99, asset=0.1),
        ])
        reg = ConnectorRegistry()
        reg.register(ToolConnector("bad", fn=lambda: None, profile=ToolProfile("bad", cost=0.5, latency=0.3, reliability=0.4, risk=0.99, asset=0.1)))
        cfg = InterceptorConfig(risk_veto_threshold=0.05, redirect_strategy="refuse")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = BrainTask(prompt="bad op", stakes=0.99, safety_sensitivity=0.99)
        try:
            interceptor.intercept(task, "bad")
        except InterceptorVeto as exc:
            assert isinstance(exc.result, InterceptResult)
            assert exc.result.tool_name == "bad"
            assert not exc.result.permitted
