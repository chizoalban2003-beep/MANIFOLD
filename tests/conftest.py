"""Shared pytest fixtures for MANIFOLD test suite."""

from __future__ import annotations

import pytest

from manifold.escalation_memory import EscalationMemory
from manifold.nervatura_world import NERVATURAWorld
from manifold.policy_learner import PolicyLearner
from manifold.policy_rules import PolicyRuleEngine


@pytest.fixture()
def small_world() -> NERVATURAWorld:
    """5×5×1 NERVATURAWorld with default CRNA values."""
    return NERVATURAWorld(5, 5, 1, default_crna=(0.4, 0.3, 1.0, 0.3))


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
