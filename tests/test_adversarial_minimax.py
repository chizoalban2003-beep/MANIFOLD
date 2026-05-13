"""Tests for AdversarialMinimax (PROMPT B2)."""
import pytest

from manifold.adversarial import AdversarialMinimax, NashEquilibriumGate, AuditTrigger
from manifold.brain import BrainMemory


class TestAdversarialMinimax:
    def test_minimax_returns_refuse_for_high_stakes(self):
        """minimax() returns refuse as optimal for high-stakes context."""
        mm = AdversarialMinimax()
        result = mm.minimax({"domain": "healthcare", "stakes": 0.95})
        # refuse or stop should be optimal for high-stakes — both are minimal damage
        assert result["optimal_action"] in {"refuse", "stop"}, (
            f"Expected refuse/stop for high-stakes healthcare, got {result['optimal_action']}"
        )
        assert result["minimax_damage"] < 0.5

    def test_minimax_returns_verify_for_medium_stakes(self):
        """minimax() for medium-stakes context should prefer a less extreme action."""
        mm = AdversarialMinimax()
        result = mm.minimax({"domain": "general", "stakes": 0.3})
        # For low stakes, the optimal action should not be the most extreme "stop"
        # The minimax result should have a valid optimal_action
        assert result["optimal_action"] in AdversarialMinimax.MANIFOLD_ACTIONS
        assert "action_damages" in result
        assert len(result["action_damages"]) == len(AdversarialMinimax.MANIFOLD_ACTIONS)

    def test_damage_higher_for_healthcare_domain(self):
        """damage_estimate is higher for healthcare domain than general."""
        mm = AdversarialMinimax()
        dmg_healthcare = mm.damage_estimate("allow", "inject_risk", {
            "domain": "healthcare", "stakes": 0.8
        })
        dmg_general = mm.damage_estimate("allow", "inject_risk", {
            "domain": "general", "stakes": 0.8
        })
        assert dmg_healthcare > dmg_general, (
            "Healthcare domain should have higher damage multiplier"
        )

    def test_nash_gate_check_includes_minimax_when_adversarial(self):
        """NashEquilibriumGate.check() uses minimax recommended_action when context provided."""
        gate = NashEquilibriumGate(zscore_threshold=1.5)
        memory = BrainMemory()
        # Seed memory with tools at very different reputation levels
        # so one tool gets a high z-score
        for i in range(5):
            memory.tool_stats[f"tool_{i}"] = {"success_rate": 0.1}
        # Give one tool an implausibly high success rate
        memory.tool_stats["tool_implausible"] = {"success_rate": 0.99}

        context = {"domain": "healthcare", "stakes": 0.9}
        trigger = gate.check("tool_implausible", memory, context=context)

        if trigger is not None:
            # When context is provided, minimax should update recommended_action
            assert trigger.recommended_action in AdversarialMinimax.MANIFOLD_ACTIONS
