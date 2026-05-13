"""Tests for EXP7 — Per-Agent Episodic Memory and Risk Calibration."""
from __future__ import annotations

import time

from manifold.agent_registry import (
    AgentRegistry,
    Episode,
    compare_assignment_quality,
)


def _registry_with_agents() -> AgentRegistry:
    """Return a registry with three agents for testing."""
    registry = AgentRegistry(stale_timeout=9999)
    registry.register("agent-a", "Finance Pro", ["finance"], "org1")
    registry.register("agent-b", "Generalist", ["general", "finance"], "org1")
    registry.register("agent-c", "Newcomer", ["finance"], "org1")
    return registry


def _ep(domain: str = "finance", risk: float = 0.3) -> Episode:
    return Episode(
        task_description="Test task",
        domain=domain,
        duration_seconds=30.0,
        success=True,
        crna_at_start={"c": 0.4, "r": 0.3, "n": 0.2, "a": 0.6},
        crna_at_end={"c": 0.4, "r": 0.2, "n": 0.1, "a": 0.5},
        risk_encountered=risk,
    )


# Test 1: record_episode stores episode in agent history
def test_record_episode_stores():
    registry = _registry_with_agents()
    ep = _ep("finance", 0.25)
    result = registry.record_episode("agent-a", ep)
    assert result is True, "record_episode should return True on success"
    rec = registry.get("agent-a")
    assert rec is not None
    assert len(rec.episode_history) == 1
    assert rec.episode_history[0].risk_encountered == 0.25


# Test 2: agent_risk_estimate returns 0.5 with no history
def test_risk_estimate_no_history():
    registry = _registry_with_agents()
    estimate = registry.agent_risk_estimate("agent-c", "finance")
    assert estimate == 0.5, f"Expected 0.5 with no history, got {estimate}"


# Test 3: agent_risk_estimate improves with domain-specific experience
def test_risk_estimate_with_experience():
    registry = _registry_with_agents()
    # Record several low-risk finance episodes
    for _ in range(5):
        registry.record_episode("agent-a", _ep("finance", 0.15))
    estimate = registry.agent_risk_estimate("agent-a", "finance")
    assert abs(estimate - 0.15) < 0.01, (
        f"Expected risk estimate ~0.15, got {estimate}"
    )
    # Episodes in a different domain should not affect finance estimate
    for _ in range(5):
        registry.record_episode("agent-a", _ep("legal", 0.9))
    estimate_finance = registry.agent_risk_estimate("agent-a", "finance")
    assert abs(estimate_finance - 0.15) < 0.01, (
        f"Other-domain episodes should not affect finance estimate: {estimate_finance}"
    )


# Test 4: best_agent_for_task selects lower-risk agent over higher-risk
def test_best_agent_selects_low_risk():
    registry = AgentRegistry(stale_timeout=9999)
    registry.register("low-risk", "Low Risk Agent", ["finance"], "org1")
    registry.register("high-risk", "High Risk Agent", ["finance"], "org1")

    # Give low-risk agent a strong safe history
    for _ in range(10):
        registry.record_episode("low-risk", _ep("finance", 0.1))

    # Give high-risk agent a dangerous history
    for _ in range(10):
        registry.record_episode("high-risk", _ep("finance", 0.9))

    best = registry.best_agent_for_task("finance")
    assert best == "low-risk", (
        f"Expected 'low-risk' agent to be selected, got {best!r}"
    )


# Test 5: compare_assignment_quality returns expected keys
def test_compare_assignment_quality():
    result = compare_assignment_quality()
    assert isinstance(result, dict)
    assert "episodic_win_rate" in result
    assert "risk_improvement" in result
    assert 0.0 <= result["episodic_win_rate"] <= 1.0
    # Episodic assignment should outperform random at least half the time
    assert result["episodic_win_rate"] >= 0.5, (
        f"Episodic win rate {result['episodic_win_rate']} should be >= 0.5"
    )


# ============================================================
# EXP7 NEW TESTS — Prompt 3 (domain_risk_estimate, agent_task_score,
#                              best_agent_for_domain, recent weighting)
# ============================================================

# New Test 1: domain_risk_estimate returns 0.5 with no history
def test_domain_risk_estimate_no_history():
    registry = _registry_with_agents()
    est = registry.domain_risk_estimate("agent-c", "finance")
    assert est == 0.5, f"Expected 0.5 with no history, got {est}"


# New Test 2: domain_risk_estimate returns lower value for low-risk specialist
def test_domain_risk_estimate_low_risk_specialist():
    registry = AgentRegistry(stale_timeout=9999)
    registry.register("specialist", "Finance Specialist", ["finance"], "org1")
    for _ in range(6):
        registry.record_episode("specialist", _ep("finance", 0.15))
    est = registry.domain_risk_estimate("specialist", "finance")
    assert est < 0.4, f"Specialist estimate should be well below 0.5, got {est}"


# New Test 3: agent_task_score ranks specialist above generalist for specialist domain
def test_agent_task_score_specialist_wins():
    registry = AgentRegistry(stale_timeout=9999)
    registry.register("spec", "Specialist", ["finance"], "org1")
    registry.register("gen", "Generalist", ["finance"], "org1")

    for _ in range(8):
        registry.record_episode("spec", _ep("finance", 0.1))
    for _ in range(8):
        registry.record_episode("gen", _ep("finance", 0.8))

    spec_score = registry.agent_task_score("spec", "finance")
    gen_score = registry.agent_task_score("gen", "finance")
    assert spec_score > gen_score, (
        f"Specialist score {spec_score:.4f} should beat generalist {gen_score:.4f}"
    )


# New Test 4: best_agent_for_domain selects specialist over newcomer with no history
def test_best_agent_for_domain_specialist_over_newcomer():
    registry = AgentRegistry(stale_timeout=9999)
    registry.register("spec", "Specialist", ["finance"], "org1")
    registry.register("new", "Newcomer", ["finance"], "org1")

    for _ in range(5):
        registry.record_episode("spec", _ep("finance", 0.15))
    # newcomer has no history (defaults to 0.5)

    best = registry.best_agent_for_domain("finance")
    assert best == "spec", f"Specialist should be selected over newcomer, got {best!r}"


# New Test 5: record_episode returns False for unknown agent
def test_record_episode_unknown_agent_returns_false():
    registry = AgentRegistry(stale_timeout=9999)
    result = registry.record_episode("nonexistent-agent", _ep())
    assert result is False, "record_episode should return False for unknown agent"


# New Test 6: Recent episodes weighted more than old ones
def test_domain_risk_estimate_recent_weighting():
    """Recent episodes (last 10) get 2× weight — switching to low risk recently
    should lower the estimate more than the old high-risk history suggests."""
    registry = AgentRegistry(stale_timeout=9999)
    registry.register("agent", "Test Agent", ["finance"], "org1")

    # 11 old high-risk episodes
    for _ in range(11):
        registry.record_episode("agent", _ep("finance", 0.9))

    # 10 recent low-risk episodes (should dominate due to 2× weighting)
    for _ in range(10):
        registry.record_episode("agent", _ep("finance", 0.1))

    est = registry.domain_risk_estimate("agent", "finance")
    # With 2× recent weighting: 1*0.9 + 2*10*0.1 = 0.9+2 = 2.9 / (1+20) = 0.138
    # Without weighting: (11*0.9 + 10*0.1) / 21 = (9.9+1)/21 = 0.519
    # The estimate should be significantly below 0.5
    assert est < 0.4, (
        f"Recent weighting should pull estimate below 0.4, got {est:.4f}"
    )

