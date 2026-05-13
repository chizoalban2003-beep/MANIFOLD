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
    registry.record_episode("agent-a", ep)
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
