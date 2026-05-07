"""Tests for Phase 14: Ecosystem Adapters (ManifoldCallbackHandler + ManifoldOpenAIWrapper)."""

from __future__ import annotations

import pytest

from manifold import (
    BrainConfig,
    ConnectorRegistry,
    GossipBus,
    InterceptorConfig,
    ManifoldBrain,
    ToolConnector,
    ToolProfile,
    default_tools,
    ActiveInterceptor,
    InterceptorVeto,
)
from manifold.adapters import ManifoldCallbackHandler, ManifoldOpenAIWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(seed: int = 42) -> ManifoldBrain:
    return ManifoldBrain(
        BrainConfig(generations=5, population_size=20, grid_size=5, seed=seed),
        default_tools(),
    )


def _make_registry() -> ConnectorRegistry:
    registry = ConnectorRegistry()
    for name in ("billing_api", "email_sender", "search_api"):
        profile = ToolProfile(name, cost=0.05, latency=0.1, reliability=0.85, risk=0.10, asset=0.70)
        registry.register(ToolConnector(name=name, fn=lambda q: {"ok": q}, profile=profile))
    return registry


class _FakeOpenAIResponse:
    """Minimal stand-in for openai.ChatCompletion response."""
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class _FakeOpenAIClient:
    """Minimal duck-typed OpenAI client for testing."""

    def __init__(self, response_content: str = "ok") -> None:
        self._content = response_content
        self.last_kwargs: dict = {}

        class _Completions:
            def create(_self, **kwargs):
                self.last_kwargs = kwargs
                return _FakeOpenAIResponse(self._content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# ---------------------------------------------------------------------------
# ManifoldCallbackHandler tests
# ---------------------------------------------------------------------------

class TestManifoldCallbackHandler:
    def test_instantiation(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        assert handler is not None

    def test_on_tool_start_records_entry(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        handler.on_tool_start({"name": "billing_api"}, "process payment")
        log = handler.call_log()
        assert len(log) == 1
        assert log[0]["tool_name"] == "billing_api"
        assert "brain_action" in log[0]

    def test_on_tool_end_marks_success(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        handler.on_tool_start({"name": "search_api"}, "find documents")
        handler.on_tool_end("results found")
        log = handler.call_log()
        assert log[0]["success"] is True
        assert "latency" in log[0]

    def test_on_tool_error_marks_failure(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        handler.on_tool_start({"name": "billing_api"}, "charge card")
        handler.on_tool_error(ValueError("timeout"))
        log = handler.call_log()
        assert log[0]["success"] is False
        assert "error" in log[0]

    def test_failure_rate_zero_initially(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        assert handler.failure_rate() == 0.0

    def test_failure_rate_calculation(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        handler.on_tool_start({"name": "t1"}, "q1")
        handler.on_tool_end("ok")
        handler.on_tool_start({"name": "t2"}, "q2")
        handler.on_tool_error(RuntimeError("boom"))
        handler.on_tool_start({"name": "t3"}, "q3")
        handler.on_tool_end("ok")
        assert abs(handler.failure_rate() - 1 / 3) < 0.01

    def test_summary_keys(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        s = handler.summary()
        assert "total_calls" in s
        assert "completed" in s
        assert "failures" in s
        assert "failure_rate" in s
        assert "avg_latency" in s

    def test_gossip_bus_integration(self):
        """Handler publishes health/failure signals to GossipBus."""
        brain = _make_brain()
        bus = GossipBus()
        handler = ManifoldCallbackHandler(brain=brain, gossip_bus=bus)
        handler.on_tool_start({"name": "search_api"}, "query")
        handler.on_tool_end("results")
        # No assertion on bus internals — just verify no exception raised
        bus.stop()

    def test_serialized_as_string(self):
        """Handler gracefully handles string instead of dict for serialized."""
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        handler.on_tool_start("billing_api", "payment")
        log = handler.call_log()
        assert log[0]["tool_name"] == "billing_api"

    def test_unknown_tool_on_error_no_crash(self):
        """on_tool_error without a preceding on_tool_start does not crash."""
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        handler.on_tool_error("some error")  # no prior on_tool_start

    def test_multiple_sequential_calls(self):
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain)
        for i in range(5):
            handler.on_tool_start({"name": f"tool_{i}"}, f"query {i}")
            handler.on_tool_end("ok")
        assert len(handler.call_log()) == 5
        assert handler.failure_rate() == 0.0

    def test_verbose_mode_does_not_crash(self, caplog):
        import logging
        brain = _make_brain()
        handler = ManifoldCallbackHandler(brain=brain, verbose=True)
        with caplog.at_level(logging.DEBUG, logger="manifold.adapters"):
            handler.on_tool_start({"name": "test_tool"}, "test input")
            handler.on_tool_end("test output")
        assert "MANIFOLD" in caplog.text


# ---------------------------------------------------------------------------
# ManifoldOpenAIWrapper tests
# ---------------------------------------------------------------------------

class TestManifoldOpenAIWrapper:
    def test_instantiation(self):
        brain = _make_brain()
        client = ManifoldOpenAIWrapper(_FakeOpenAIClient(), brain=brain)
        assert client is not None

    def test_chat_completions_create_forwards(self):
        brain = _make_brain()
        fake = _FakeOpenAIClient(response_content="hello")
        wrapper = ManifoldOpenAIWrapper(fake, brain=brain, domain="general")
        resp = wrapper.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        assert resp.choices[0].message.content == "hello"

    def test_call_log_populated(self):
        brain = _make_brain()
        fake = _FakeOpenAIClient()
        wrapper = ManifoldOpenAIWrapper(fake, brain=brain)
        wrapper.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        log = wrapper.call_log()
        assert len(log) == 1
        assert log[0]["model"] == "gpt-4o-mini"
        assert "estimated_stakes" in log[0]

    def test_summary_keys(self):
        brain = _make_brain()
        wrapper = ManifoldOpenAIWrapper(_FakeOpenAIClient(), brain=brain)
        s = wrapper.summary()
        assert "total_calls" in s
        assert "avg_estimated_stakes" in s
        assert "avg_estimated_complexity" in s

    def test_summary_empty(self):
        brain = _make_brain()
        wrapper = ManifoldOpenAIWrapper(_FakeOpenAIClient(), brain=brain)
        s = wrapper.summary()
        assert s["total_calls"] == 0
        assert s["avg_estimated_stakes"] == 0.0

    def test_interceptor_permits_low_risk(self):
        """Low-risk calls pass through the interceptor."""
        brain = _make_brain()
        registry = _make_registry()
        interceptor = ActiveInterceptor(
            registry=registry,
            brain=brain,
            config=InterceptorConfig(risk_veto_threshold=0.95),  # very high threshold
        )
        fake = _FakeOpenAIClient()
        wrapper = ManifoldOpenAIWrapper(
            fake, brain=brain, interceptor=interceptor, registry=registry, domain="general"
        )
        # billing_api has risk=0.10, so stakes*risk should be well below 0.95
        resp = wrapper.chat.completions.create(
            model="billing_api",
            messages=[{"role": "user", "content": "list invoices"}],
        )
        # Should not raise; result returned
        assert resp is not None

    def test_interceptor_vetos_high_risk(self):
        """High-risk calls are blocked by the interceptor."""
        brain = _make_brain(seed=99)
        registry = ConnectorRegistry()
        # Create a very risky tool
        risky_profile = ToolProfile("risky_tool", cost=0.5, latency=0.5, reliability=0.5, risk=0.95, asset=0.2)
        registry.register(ToolConnector(name="risky_tool", fn=lambda q: q, profile=risky_profile))
        interceptor = ActiveInterceptor(
            registry=registry,
            brain=brain,
            config=InterceptorConfig(risk_veto_threshold=0.01),  # extremely strict
        )
        fake = _FakeOpenAIClient()
        wrapper = ManifoldOpenAIWrapper(
            fake, brain=brain, interceptor=interceptor, registry=registry, domain="finance"
        )
        with pytest.raises(InterceptorVeto):
            wrapper.chat.completions.create(
                model="risky_tool",
                messages=[{"role": "user", "content": "transfer all funds immediately"}],
            )

    def test_auto_registers_unknown_model(self):
        """Wrapper auto-registers models not in the registry."""
        brain = _make_brain()
        registry = ConnectorRegistry()
        interceptor = ActiveInterceptor(
            registry=registry,
            brain=brain,
            config=InterceptorConfig(risk_veto_threshold=0.99),  # very permissive
        )
        fake = _FakeOpenAIClient()
        wrapper = ManifoldOpenAIWrapper(
            fake, brain=brain, interceptor=interceptor, registry=registry
        )
        wrapper.chat.completions.create(
            model="new-model-xyz",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert registry.get("new-model-xyz") is not None

    def test_no_brain_uses_default(self):
        """Wrapper works without explicit brain (uses BrainConfig default)."""
        fake = _FakeOpenAIClient()
        wrapper = ManifoldOpenAIWrapper(fake)  # no brain= kwarg
        resp = wrapper.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert resp is not None

    def test_non_user_messages_handled(self):
        """Wrapper does not crash when messages have no user role."""
        brain = _make_brain()
        wrapper = ManifoldOpenAIWrapper(_FakeOpenAIClient(), brain=brain)
        wrapper.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}],
        )
        log = wrapper.call_log()
        assert len(log) == 1

    def test_multiple_calls_accumulate_log(self):
        brain = _make_brain()
        wrapper = ManifoldOpenAIWrapper(_FakeOpenAIClient(), brain=brain)
        for i in range(3):
            wrapper.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"message {i}"}],
            )
        assert len(wrapper.call_log()) == 3
        s = wrapper.summary()
        assert s["total_calls"] == 3
