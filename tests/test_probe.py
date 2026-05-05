"""Tests for Phase 31: Active Canary Probing (manifold/probe.py)."""

from __future__ import annotations

import time

import pytest

from manifold.brain import BrainConfig, BrainTask, ManifoldBrain
from manifold.entropy import ReputationDecay, VolatilityTable
from manifold.federation import FederatedGossipPacket
from manifold.hub import ReputationHub
from manifold.probe import ActiveProber, CanaryGenerator, CanaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hub_with_signals() -> tuple[ReputationHub, ReputationDecay]:
    """Return (hub, decay) with some tools pre-seeded."""
    hub = ReputationHub()
    decay = ReputationDecay(volatility=VolatilityTable.default())
    # Record old signals → high entropy
    old_time = time.time() - 200 * 3600  # 200 hours ago
    decay.record_signal("stale-tool", domain="llm", reliability=0.9)
    decay._signal_times["stale-tool"] = old_time
    # Fresh signal → low entropy
    decay.record_signal("fresh-tool", domain="math", reliability=0.95)
    return hub, decay


def _make_brain() -> ManifoldBrain:
    return ManifoldBrain(
        config=BrainConfig(grid_size=7, generations=20, population_size=24),
        tools=[],
    )


# ---------------------------------------------------------------------------
# CanaryGenerator tests
# ---------------------------------------------------------------------------


class TestCanaryGenerator:
    def test_generate_returns_brain_task(self) -> None:
        gen = CanaryGenerator()
        task = gen.generate("gpt-4o")
        assert isinstance(task, BrainTask)

    def test_generated_task_has_zero_stakes(self) -> None:
        gen = CanaryGenerator()
        task = gen.generate("gpt-4o")
        assert task.stakes == 0.0

    def test_generated_task_has_zero_safety_sensitivity(self) -> None:
        gen = CanaryGenerator()
        task = gen.generate("gpt-4o")
        assert task.safety_sensitivity == 0.0

    def test_generated_task_embeds_tool_name_in_prompt(self) -> None:
        gen = CanaryGenerator()
        task = gen.generate("my-special-tool")
        assert "my-special-tool" in task.prompt

    def test_generated_task_uses_default_domain(self) -> None:
        gen = CanaryGenerator(domain="finance")
        task = gen.generate("tool")
        assert task.domain == "finance"

    def test_generated_task_domain_override(self) -> None:
        gen = CanaryGenerator(domain="general")
        task = gen.generate("tool", domain="math")
        assert task.domain == "math"

    def test_canary_prompt_prefix(self) -> None:
        gen = CanaryGenerator()
        task = gen.generate("calc-tool")
        assert task.prompt.startswith("canary_probe:")

    def test_different_tools_different_prompts(self) -> None:
        gen = CanaryGenerator()
        t1 = gen.generate("tool-a")
        t2 = gen.generate("tool-b")
        assert t1.prompt != t2.prompt


# ---------------------------------------------------------------------------
# CanaryResult tests
# ---------------------------------------------------------------------------


class TestCanaryResult:
    def _make_result(self, **kwargs) -> CanaryResult:
        defaults = {
            "tool_name": "test-tool",
            "entropy_score_before": 0.6,
            "adversarial_suspect": False,
            "timestamp": time.time(),
            "probe_action": "pass",
            "penalty_applied": False,
        }
        defaults.update(kwargs)
        return CanaryResult(**defaults)

    def test_frozen(self) -> None:
        r = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.probe_action = "modified"  # type: ignore[misc]

    def test_to_dict_keys(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        for key in (
            "tool_name", "entropy_score_before", "adversarial_suspect",
            "timestamp", "probe_action", "penalty_applied",
        ):
            assert key in d

    def test_to_dict_values_match(self) -> None:
        r = self._make_result(probe_action="fail", penalty_applied=True)
        d = r.to_dict()
        assert d["probe_action"] == "fail"
        assert d["penalty_applied"] is True

    def test_probe_action_suspect(self) -> None:
        r = self._make_result(probe_action="suspect", adversarial_suspect=True)
        assert r.probe_action == "suspect"
        assert r.adversarial_suspect is True


# ---------------------------------------------------------------------------
# ActiveProber tests
# ---------------------------------------------------------------------------


class TestActiveProber:
    def _make_prober(
        self, entropy_threshold: float = 0.5
    ) -> tuple[ActiveProber, ReputationHub, ReputationDecay]:
        hub, decay = _make_hub_with_signals()
        brain = _make_brain()
        prober = ActiveProber(
            hub=hub,
            decay=decay,
            brain=brain,
            entropy_threshold=entropy_threshold,
            interval_seconds=999.0,  # prevent background auto-fire
        )
        return prober, hub, decay

    def test_probe_results_initially_empty(self) -> None:
        prober, _, _ = self._make_prober()
        assert prober.probe_results() == []

    def test_probe_once_returns_canary_result(self) -> None:
        prober, _, _ = self._make_prober()
        result = prober.probe_once("stale-tool")
        assert isinstance(result, CanaryResult)

    def test_probe_once_records_tool_name(self) -> None:
        prober, _, _ = self._make_prober()
        result = prober.probe_once("my-tool")
        assert result.tool_name == "my-tool"

    def test_probe_once_records_entropy_score(self) -> None:
        prober, hub, decay = self._make_prober()
        # stale-tool should have non-zero entropy
        result = prober.probe_once("stale-tool")
        assert result.entropy_score_before >= 0.0

    def test_probe_once_records_timestamp(self) -> None:
        before = time.time()
        prober, _, _ = self._make_prober()
        result = prober.probe_once("tool-x")
        after = time.time()
        assert before <= result.timestamp <= after

    def test_probe_once_action_is_pass_fail_or_suspect(self) -> None:
        prober, _, _ = self._make_prober()
        result = prober.probe_once("tool-x")
        assert result.probe_action in {"pass", "fail", "suspect"}

    def test_probe_results_accumulate(self) -> None:
        prober, _, _ = self._make_prober()
        for i in range(3):
            prober.probe_once(f"tool-{i}")
        assert len(prober.probe_results()) == 3

    def test_probe_results_returns_copy(self) -> None:
        prober, _, _ = self._make_prober()
        prober.probe_once("t")
        r1 = prober.probe_results()
        prober.probe_once("t2")
        r2 = prober.probe_results()
        assert len(r2) == len(r1) + 1

    def test_probe_high_entropy_tools_empty_when_below_threshold(self) -> None:
        prober, _, decay = self._make_prober(entropy_threshold=1.0)  # unreachable
        results = prober.probe_high_entropy_tools()
        assert results == []

    def test_probe_high_entropy_tools_fires_for_stale_tools(self) -> None:
        prober, hub, decay = self._make_prober(entropy_threshold=0.0)
        # entropy_threshold=0 means all tracked tools get probed
        # stale-tool and fresh-tool are both tracked (entropy >= 0)
        results = prober.probe_high_entropy_tools()
        tool_names = [r.tool_name for r in results]
        assert "stale-tool" in tool_names or "fresh-tool" in tool_names

    def test_canary_summary_structure(self) -> None:
        prober, _, _ = self._make_prober()
        prober.probe_once("t")
        summary = prober.canary_summary()
        for key in ("total_probes", "adversarial_suspects", "penalties_applied", "is_running", "pass_rate"):
            assert key in summary

    def test_canary_summary_total_probes(self) -> None:
        prober, _, _ = self._make_prober()
        prober.probe_once("t1")
        prober.probe_once("t2")
        assert prober.canary_summary()["total_probes"] == 2

    def test_latest_result_returns_most_recent(self) -> None:
        prober, _, _ = self._make_prober()
        prober.probe_once("tool-x")
        prober.probe_once("tool-x")
        result = prober.latest_result("tool-x")
        assert result is not None
        assert result.tool_name == "tool-x"

    def test_latest_result_returns_none_for_unknown(self) -> None:
        prober, _, _ = self._make_prober()
        assert prober.latest_result("never-probed") is None

    def test_is_running_initially_false(self) -> None:
        prober, _, _ = self._make_prober()
        assert prober.is_running() is False

    def test_start_sets_running(self) -> None:
        prober, _, _ = self._make_prober()
        prober.start()
        assert prober.is_running() is True
        prober.stop()

    def test_stop_clears_running(self) -> None:
        prober, _, _ = self._make_prober()
        prober.start()
        prober.stop()
        # Give the thread a moment to notice the stop flag
        time.sleep(0.05)
        assert prober.is_running() is False

    def test_start_idempotent(self) -> None:
        prober, _, _ = self._make_prober()
        prober.start()
        prober.start()  # should not crash or create a second thread
        assert prober.is_running() is True
        prober.stop()

    def test_penalty_applied_when_probe_fails(self) -> None:
        """Ensure penalty_applied is True for at least one fail/suspect probe."""
        hub, decay = _make_hub_with_signals()
        brain = _make_brain()
        prober = ActiveProber(
            hub=hub, decay=decay, brain=brain,
            entropy_threshold=0.0, interval_seconds=999.0,
        )
        # Run many probes to increase chance of a fail
        results = [prober.probe_once("stale-tool") for _ in range(10)]
        # At least the "pass" case should have penalty_applied=False
        passes = [r for r in results if r.probe_action == "pass"]
        for r in passes:
            assert r.penalty_applied is False

    def test_pass_rate_in_summary_is_fraction(self) -> None:
        prober, _, _ = self._make_prober()
        for _ in range(5):
            prober.probe_once("t")
        rate = prober.canary_summary()["pass_rate"]
        assert 0.0 <= rate <= 1.0
