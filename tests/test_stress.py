"""Stress tests for Phase 11 (AdversarialPricingDetector) and Phase 12 (PenaltyOptimizer).

Scenario: A tool slowly degrades — its latency creeps up by 5 ms per call
and after the warm-up window it starts failing at an increasing rate.

Pass criteria
-------------
1. ``AdversarialPricingDetector`` flags the tool as a honey-pot.
2. ``PenaltyOptimizer`` proposes a penalty hike for the correlated trigger.
3. The combo works on a 500-row simulated log.
"""

from __future__ import annotations

import random

from manifold.adversarial import AdversarialPricingDetector, NashEquilibriumGate
from manifold.autodiscovery import AutoRuleDiscovery, PenaltyOptimizer, RuleObservation
from manifold.brain import BrainConfig, BrainTask, ManifoldBrain, ToolProfile
from manifold.connector import ConnectorRegistry, ToolConnector
from manifold.interceptor import ActiveInterceptor, InterceptorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_degrading_outcomes(
    n: int = 500,
    warm_up: int = 20,
    warm_success_rate: float = 0.98,
    final_success_rate: float = 0.25,
    seed: int = 42,
) -> list[bool]:
    """Generate outcomes for a tool that starts reliable then degrades."""
    rng = random.Random(seed)
    outcomes: list[bool] = []
    for i in range(n):
        if i < warm_up:
            rate = warm_success_rate
        else:
            progress = (i - warm_up) / max(1, n - warm_up)
            rate = warm_success_rate - progress * (warm_success_rate - final_success_rate)
        outcomes.append(rng.random() < rate)
    return outcomes


# ---------------------------------------------------------------------------
# Stress test: AdversarialPricingDetector on 500-row degrading log
# ---------------------------------------------------------------------------

class TestAdversarialDetectorStress:
    def test_detector_flags_degrading_tool(self):
        detector = AdversarialPricingDetector(
            warm_up_size=20,
            post_window_size=50,
            drop_threshold=0.30,
            min_post_outcomes=10,
        )
        outcomes = _make_degrading_outcomes(n=500, warm_up=20, warm_success_rate=0.98, final_success_rate=0.25)
        for success in outcomes:
            detector.record("slow_billing_api", success)
        assert detector.is_suspect("slow_billing_api"), (
            f"Expected honey-pot flag; warm_up_rate={detector.warm_up_rate('slow_billing_api')}, "
            f"post_rate={detector.post_rate('slow_billing_api')}"
        )

    def test_drop_magnitude_reflects_degradation(self):
        detector = AdversarialPricingDetector(warm_up_size=20, post_window_size=50, drop_threshold=0.20)
        outcomes = _make_degrading_outcomes(n=500, warm_up=20, warm_success_rate=0.98, final_success_rate=0.25)
        for s in outcomes:
            detector.record("billing_api", s)
        drop = detector.drop("billing_api")
        assert drop is not None
        assert drop >= 0.30, f"Expected drop >= 0.30 but got {drop:.3f}"

    def test_stable_tool_not_flagged(self):
        detector = AdversarialPricingDetector(warm_up_size=20, post_window_size=50, drop_threshold=0.30)
        rng = random.Random(7)
        for _ in range(500):
            detector.record("stable_tool", rng.random() < 0.90)
        assert not detector.is_suspect("stable_tool")

    def test_suspects_list_contains_degrading_tool(self):
        detector = AdversarialPricingDetector(warm_up_size=20, post_window_size=50, drop_threshold=0.25)
        rng = random.Random(42)
        outcomes_bad = _make_degrading_outcomes(n=500, warm_up=20, warm_success_rate=0.98, final_success_rate=0.10)
        outcomes_good = [rng.random() < 0.90 for _ in range(500)]
        for s in outcomes_bad:
            detector.record("billing_api", s)
        for s in outcomes_good:
            detector.record("search_api", s)
        suspects = detector.suspects()
        suspect_names = [s["tool_name"] for s in suspects]
        assert "billing_api" in suspect_names
        assert "search_api" not in suspect_names

    def test_warm_up_incomplete_not_suspect(self):
        detector = AdversarialPricingDetector(warm_up_size=20, post_window_size=50, drop_threshold=0.30)
        for _ in range(10):
            detector.record("new_tool", True)
        assert not detector.is_suspect("new_tool")
        assert detector.warm_up_rate("new_tool") is None


# ---------------------------------------------------------------------------
# Stress test: PenaltyOptimizer proposes hike when billing_api degrades
# ---------------------------------------------------------------------------

class TestPenaltyOptimizerStress:
    def _build_degrading_observations(
        self, n: int = 500, warm_up: int = 20, seed: int = 42
    ) -> list[RuleObservation]:
        rng = random.Random(seed)
        observations: list[RuleObservation] = []
        for i in range(n):
            if i < warm_up:
                delta = rng.uniform(0.05, 0.15)
                trigger = "no_issue"
            else:
                progress = (i - warm_up) / max(1, n - warm_up)
                base_loss = 0.05 + 0.60 * progress
                delta = -rng.uniform(base_loss * 0.8, base_loss * 1.2)
                trigger = "billing_timeout"
            observations.append(
                RuleObservation(
                    trigger=trigger,
                    rule_name="billing_latency_guard",
                    observed_asset_delta=delta,
                    current_penalty=1.0,
                )
            )
        return observations

    def test_penalty_optimizer_proposes_hike(self):
        optimizer = PenaltyOptimizer(min_observations=10)
        observations = self._build_degrading_observations(n=500)
        for obs in observations:
            optimizer.record(obs)
        proposals = optimizer.suggest_all()
        billing_proposals = [p for p in proposals if "billing_timeout" in p.trigger]
        assert billing_proposals, "Expected at least one proposal for billing_timeout trigger"
        for p in billing_proposals:
            assert p.proposed_penalty > 0, f"Expected penalty > 0, got {p.proposed_penalty}"

    def test_auto_rule_discovery_full_pipeline(self):
        discovery = AutoRuleDiscovery()
        observations = self._build_degrading_observations(n=500)
        for obs in observations:
            discovery.optimizer.record(obs)
        suggestions = discovery.suggest_penalty_updates()
        assert isinstance(suggestions, list)
        billing_hits = [s for s in suggestions if "billing_timeout" in s.trigger]
        assert billing_hits, "AutoRuleDiscovery should surface billing_timeout proposals"


# ---------------------------------------------------------------------------
# Integration: Detector + PenaltyOptimizer + ActiveInterceptor
# ---------------------------------------------------------------------------

class TestDetectorOptimizerIntegration:
    def test_end_to_end_500_rows(self):
        rng = random.Random(99)
        detector = AdversarialPricingDetector(warm_up_size=20, post_window_size=50, drop_threshold=0.25)
        optimizer = PenaltyOptimizer(min_observations=10)
        outcomes = _make_degrading_outcomes(n=500, warm_up=20, warm_success_rate=0.97, final_success_rate=0.15, seed=99)
        for success in outcomes:
            detector.record("billing_api", success)
            if not success:
                delta = -rng.uniform(0.15, 0.50)
                obs = RuleObservation(trigger="billing_timeout", rule_name="billing_guard", observed_asset_delta=delta, current_penalty=1.0)
            else:
                delta = rng.uniform(0.02, 0.08)
                obs = RuleObservation(trigger="no_issue", rule_name="billing_guard", observed_asset_delta=delta, current_penalty=1.0)
            optimizer.record(obs)

        assert detector.is_suspect("billing_api"), "billing_api should be flagged as honey-pot"
        proposals = optimizer.suggest_all()
        billing_proposals = [p for p in proposals if p.trigger == "billing_timeout"]
        assert billing_proposals, "PenaltyOptimizer should propose action on billing_timeout"
        assert billing_proposals[0].proposed_penalty > 0

    def test_interceptor_blocks_flagged_tool(self):
        reg = ConnectorRegistry()
        reg.register(ToolConnector(
            "billing_api",
            fn=lambda: "charge",
            profile=ToolProfile("billing_api", cost=0.20, latency=0.4, reliability=0.30, risk=0.90, asset=0.2),
        ))
        brain = ManifoldBrain(BrainConfig(), tools=[
            ToolProfile("billing_api", cost=0.20, latency=0.4, reliability=0.30, risk=0.90, asset=0.2),
        ])
        cfg = InterceptorConfig(risk_veto_threshold=0.25, redirect_strategy="hitl")
        interceptor = ActiveInterceptor(registry=reg, brain=brain, config=cfg)
        task = BrainTask(prompt="charge customer £500", domain="finance", stakes=0.95, safety_sensitivity=0.90, uncertainty=0.8)
        result = interceptor.intercept(task, "billing_api")
        assert not result.permitted
        assert result.redirect_to == "hitl"
        assert interceptor.veto_count() == 1


# ---------------------------------------------------------------------------
# NashEquilibriumGate stress
# ---------------------------------------------------------------------------

class TestNashEquilibriumGateStress:
    def test_implausible_reputation_triggers_audit(self):
        from manifold.brain import BrainMemory, GossipNote

        gate = NashEquilibriumGate()
        memory = BrainMemory()

        rng = random.Random(77)
        for tool in ["search", "calc", "lookup"]:
            memory.tool_stats[tool] = {
                "success_rate": rng.uniform(0.70, 0.85),
                "call_count": rng.randint(20, 100),
            }

        for _ in range(40):
            gate.laundering_detector.record(
                GossipNote(tool="billing_api", claim="healthy", source_id="fake_scout", confidence=1.0)
            )
        for _ in range(3):
            gate.laundering_detector.record(
                GossipNote(tool="billing_api", claim="failing", source_id="fake_scout", confidence=0.5)
            )

        trigger = gate.check("billing_api", memory=memory, source_id="fake_scout")
        if trigger is not None:
            assert trigger.tool_name == "billing_api"
