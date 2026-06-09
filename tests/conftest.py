"""Shared pytest fixtures for MANIFOLD test suite."""

from __future__ import annotations

import pytest

from manifold.escalation_memory import EscalationMemory
from manifold.policy_learner import PolicyLearner
from manifold.policy_rules import PolicyRuleEngine


@pytest.fixture()
def fresh_memory() -> EscalationMemory:
    """EscalationMemory with low confidence threshold for fast tests."""
    return EscalationMemory(confidence_threshold=0.85, min_decisions=3)


@pytest.fixture()
def fresh_registry() -> PolicyRuleEngine:
    """Empty PolicyRuleEngine."""
    return PolicyRuleEngine()


@pytest.fixture()
def fresh_learner(fresh_memory: EscalationMemory, fresh_registry: PolicyRuleEngine) -> PolicyLearner:
    """PolicyLearner wired to fresh memory and registry."""
    return PolicyLearner(
        memory=fresh_memory,
        registry=fresh_registry,
        promote_threshold=0.9,
    )
