"""Tests for Phase 11: Adversarial Robustness & Nash Equilibrium Gate."""

import pytest

from manifold.adversarial import (
    AdversarialPricingDetector,
    AuditTrigger,
    NashEquilibriumGate,
    ReputationLaunderingDetector,
    ToolOutcomeWindow,
)
from manifold.brain import BrainMemory, GossipNote


# ---------------------------------------------------------------------------
# ToolOutcomeWindow
# ---------------------------------------------------------------------------


def test_window_empty_returns_1():
    w = ToolOutcomeWindow(window_size=10)
    assert w.recent_success_rate() == 1.0


def test_window_all_success():
    w = ToolOutcomeWindow(window_size=10)
    for _ in range(5):
        w.record(True)
    assert w.recent_success_rate() == 1.0


def test_window_all_failure():
    w = ToolOutcomeWindow(window_size=10)
    for _ in range(5):
        w.record(False)
    assert w.recent_success_rate() == 0.0


def test_window_mixed():
    w = ToolOutcomeWindow(window_size=10)
    for _ in range(3):
        w.record(True)
    for _ in range(1):
        w.record(False)
    assert abs(w.recent_success_rate() - 0.75) < 1e-6


def test_window_evicts_oldest():
    w = ToolOutcomeWindow(window_size=3)
    w.record(False)
    w.record(False)
    w.record(False)
    w.record(True)  # evicts first False
    # now: [False, False, True] -> 1/3
    assert abs(w.recent_success_rate() - 1 / 3) < 1e-6


def test_window_last_n():
    w = ToolOutcomeWindow(window_size=20)
    for _ in range(10):
        w.record(True)
    for _ in range(10):
        w.record(False)
    # last 5 should all be False
    assert w.recent_success_rate(last_n=5) == 0.0
    # last 15 should mix
    assert 0 < w.recent_success_rate(last_n=15) < 1


def test_window_count():
    w = ToolOutcomeWindow(window_size=5)
    for _ in range(4):
        w.record(True)
    assert w.count() == 4


def test_window_clear():
    w = ToolOutcomeWindow(window_size=5)
    for _ in range(5):
        w.record(True)
    w.clear()
    assert w.count() == 0


# ---------------------------------------------------------------------------
# ReputationLaunderingDetector
# ---------------------------------------------------------------------------


def _make_gossip_note(source_id: str, tool: str, claim: str) -> GossipNote:
    return GossipNote(
        source_id=source_id,
        tool=tool,
        claim=claim,
        source_reputation=0.7,
    )


def test_laundering_no_data_not_suspect():
    d = ReputationLaunderingDetector(boost_threshold=0.85, min_claims=10)
    assert not d.is_suspect("src", "tool")


def test_laundering_below_min_claims_not_suspect():
    d = ReputationLaunderingDetector(boost_threshold=0.85, min_claims=10)
    for _ in range(9):
        d.record(_make_gossip_note("src", "tool", "healthy"))
    assert not d.is_suspect("src", "tool")


def test_laundering_detects_boosting():
    d = ReputationLaunderingDetector(boost_threshold=0.85, min_claims=10)
    for _ in range(11):
        d.record(_make_gossip_note("src", "tool", "healthy"))
    assert d.is_suspect("src", "tool")


def test_laundering_mixed_not_suspect():
    d = ReputationLaunderingDetector(boost_threshold=0.85, min_claims=10)
    for _ in range(5):
        d.record(_make_gossip_note("src", "tool", "healthy"))
    for _ in range(6):
        d.record(_make_gossip_note("src", "tool", "failing"))
    assert not d.is_suspect("src", "tool")


def test_laundering_boost_fraction_correct():
    d = ReputationLaunderingDetector()
    for _ in range(8):
        d.record(_make_gossip_note("src", "tool", "healthy"))
    for _ in range(2):
        d.record(_make_gossip_note("src", "tool", "failing"))
    bf = d.boost_fraction("src", "tool")
    assert abs(bf - 0.8) < 1e-6


def test_laundering_suspects_list():
    d = ReputationLaunderingDetector(boost_threshold=0.85, min_claims=5)
    for _ in range(6):
        d.record(_make_gossip_note("bad_src", "tool_x", "healthy"))
    suspects = d.suspects()
    assert len(suspects) == 1
    assert suspects[0]["source_id"] == "bad_src"
    assert suspects[0]["tool_name"] == "tool_x"


def test_laundering_total_claims():
    d = ReputationLaunderingDetector()
    for _ in range(7):
        d.record(_make_gossip_note("src", "tool", "healthy"))
    assert d.total_claims("src", "tool") == 7


# ---------------------------------------------------------------------------
# AdversarialPricingDetector
# ---------------------------------------------------------------------------


def test_pricing_no_data_not_suspect():
    d = AdversarialPricingDetector()
    assert not d.is_suspect("tool")


def test_pricing_warm_up_incomplete():
    d = AdversarialPricingDetector(warm_up_size=5)
    for _ in range(4):
        d.record("tool", True)
    assert d.warm_up_rate("tool") is None
    assert not d.is_suspect("tool")


def test_pricing_honey_pot_detected():
    d = AdversarialPricingDetector(
        warm_up_size=10, post_window_size=20, drop_threshold=0.4, min_post_outcomes=5
    )
    # Excellent warm-up
    for _ in range(10):
        d.record("tool", True)
    # Bad post-warm-up
    for _ in range(10):
        d.record("tool", False)
    assert d.is_suspect("tool")


def test_pricing_no_honey_pot_when_consistently_good():
    d = AdversarialPricingDetector(
        warm_up_size=10, post_window_size=20, drop_threshold=0.4, min_post_outcomes=5
    )
    for _ in range(20):
        d.record("tool", True)
    assert not d.is_suspect("tool")


def test_pricing_no_honey_pot_insufficient_post_data():
    d = AdversarialPricingDetector(
        warm_up_size=10, post_window_size=20, drop_threshold=0.4, min_post_outcomes=5
    )
    for _ in range(10):
        d.record("tool", True)
    for _ in range(4):  # only 4, need 5
        d.record("tool", False)
    assert not d.is_suspect("tool")


def test_pricing_drop_computed():
    d = AdversarialPricingDetector(warm_up_size=5, min_post_outcomes=5)
    for _ in range(5):
        d.record("tool", True)
    for _ in range(5):
        d.record("tool", False)
    drop = d.drop("tool")
    assert drop is not None
    assert abs(drop - 1.0) < 0.01  # warm_up=1.0, post=0.0


def test_pricing_suspects_list():
    d = AdversarialPricingDetector(
        warm_up_size=5, post_window_size=10, drop_threshold=0.3, min_post_outcomes=5
    )
    for _ in range(5):
        d.record("bad_tool", True)
    for _ in range(8):
        d.record("bad_tool", False)
    suspects = d.suspects()
    assert any(s["tool_name"] == "bad_tool" for s in suspects)


# ---------------------------------------------------------------------------
# NashEquilibriumGate
# ---------------------------------------------------------------------------


def _populated_memory() -> BrainMemory:
    mem = BrainMemory()
    # Populate several tools with varying success rates
    for tool, sr in [("a", 0.5), ("b", 0.6), ("c", 0.55), ("d", 0.58)]:
        mem.tool_stats[tool] = {"success_rate": sr, "count": 20.0, "utility": 0.0, "consecutive_failures": 0.0}
    return mem


def test_gate_no_trigger_for_normal_tool():
    gate = NashEquilibriumGate(zscore_threshold=2.0)
    mem = _populated_memory()
    trigger = gate.check("a", mem)
    assert trigger is None


def test_gate_zscore_trigger_for_implausible_rep():
    gate = NashEquilibriumGate(zscore_threshold=1.5)
    mem = _populated_memory()
    # Add a tool with implausibly high reputation
    mem.tool_stats["superstar"] = {"success_rate": 0.99, "count": 5.0, "utility": 0.5, "consecutive_failures": 0.0}
    trigger = gate.check("superstar", mem)
    assert trigger is not None
    assert trigger.trigger_type == "implausible_rep"
    assert trigger.tool_name == "superstar"
    assert trigger.z_score > 1.5


def test_gate_laundering_trigger_takes_priority():
    gate = NashEquilibriumGate(zscore_threshold=2.0)
    for _ in range(12):
        gate.laundering_detector.record(_make_gossip_note("bad_src", "tool", "healthy"))
    mem = _populated_memory()
    trigger = gate.check("tool", mem, source_id="bad_src")
    assert trigger is not None
    assert trigger.trigger_type == "laundering"


def test_gate_honeypot_trigger():
    gate = NashEquilibriumGate()
    d = gate.pricing_detector
    d.warm_up_size = 5
    d.min_post_outcomes = 5
    d.drop_threshold = 0.4
    for _ in range(5):
        d.record("target_tool", True)
    for _ in range(5):
        d.record("target_tool", False)
    mem = _populated_memory()
    trigger = gate.check("target_tool", mem)
    assert trigger is not None
    assert trigger.trigger_type == "honeypot"


def test_gate_check_all_returns_list():
    gate = NashEquilibriumGate(zscore_threshold=1.0)
    mem = _populated_memory()
    mem.tool_stats["outlier"] = {"success_rate": 0.99, "count": 10.0, "utility": 0.5, "consecutive_failures": 0.0}
    triggers = gate.check_all(mem)
    assert isinstance(triggers, list)
    tool_names = [t.tool_name for t in triggers]
    assert "outlier" in tool_names


def test_gate_zscore_returns_none_for_unknown_tool():
    gate = NashEquilibriumGate()
    mem = BrainMemory()
    assert gate.zscore("unknown", mem) is None


def test_gate_zscore_zero_std():
    gate = NashEquilibriumGate()
    mem = BrainMemory()
    mem.tool_stats["t1"] = {"success_rate": 0.8, "count": 5.0, "utility": 0.0, "consecutive_failures": 0.0}
    # Only one tool, std=0
    z = gate.zscore("t1", mem)
    assert z == 0.0


def test_audit_trigger_fields():
    trigger = AuditTrigger(
        tool_name="some_tool",
        reason="reason",
        trigger_type="honeypot",
        z_score=2.5,
    )
    assert trigger.recommended_action == "deploy_audit_scout"
    assert trigger.trigger_type == "honeypot"
