"""Tests for PolicyLearner domain-specific minimum decision thresholds (Fix 8.5)."""

from __future__ import annotations

import uuid

import pytest

from manifold.escalation_memory import EscalationMemory, EscalationRecord
from manifold.policy_learner import DOMAIN_MIN_DECISIONS, PolicyLearner
from manifold.policy_rules import PolicyRuleEngine


def _make_record(domain: str, decision: str = "approve") -> EscalationRecord:
    return EscalationRecord(
        escalation_id=str(uuid.uuid4()),
        agent_id="test-agent",
        action="execute",
        domain=domain,
        risk_score=0.65,
        context_hash=f"hash-{domain}",
        human_decision=decision,
    )


def _learner_with_domain(domain: str, n_approve: int) -> tuple[PolicyLearner, str]:
    memory = EscalationMemory(confidence_threshold=0.85, min_decisions=3)
    registry = PolicyRuleEngine()
    learner = PolicyLearner(memory=memory, registry=registry, promote_threshold=0.9)
    ctx = f"hash-{domain}"
    for _ in range(n_approve):
        memory.record(_make_record(domain, "approve"))
    return learner, ctx


class TestDomainMinDecisions:
    def test_domain_mins_defined(self):
        assert DOMAIN_MIN_DECISIONS["healthcare"] == 50
        assert DOMAIN_MIN_DECISIONS["legal"] == 50
        assert DOMAIN_MIN_DECISIONS["devops"] == 5
        assert DOMAIN_MIN_DECISIONS["finance"] == 3

    def test_healthcare_blocked_below_50(self):
        learner, ctx = _learner_with_domain("healthcare", 10)
        rule = learner.promote_to_rule(ctx)
        assert rule is None, "healthcare must not promote with only 10 decisions"

    def test_healthcare_promotes_at_50(self):
        learner, ctx = _learner_with_domain("healthcare", 50)
        rule = learner.promote_to_rule(ctx)
        assert rule is not None, "healthcare must promote at 50 consistent decisions"
        assert rule.action == "allow"

    def test_devops_promotes_quickly(self):
        learner, ctx = _learner_with_domain("devops", 5)
        rule = learner.promote_to_rule(ctx)
        assert rule is not None, "devops must promote after only 5 decisions"

    def test_legal_blocked_below_50(self):
        learner, ctx = _learner_with_domain("legal", 20)
        rule = learner.promote_to_rule(ctx)
        assert rule is None

    def test_min_for_domain_returns_default_for_unknown(self):
        memory = EscalationMemory(confidence_threshold=0.85, min_decisions=3)
        registry = PolicyRuleEngine()
        learner = PolicyLearner(memory=memory, registry=registry, default_min_decisions=7)
        assert learner._min_for_domain("unknown_domain") == 7

    def test_caller_override_is_respected(self):
        memory = EscalationMemory(confidence_threshold=0.85, min_decisions=3)
        registry = PolicyRuleEngine()
        learner = PolicyLearner(
            memory=memory,
            registry=registry,
            domain_min_decisions={"devops": 20},
        )
        assert learner._min_for_domain("devops") == 20

    def test_effective_min_is_stricter_of_global_and_domain(self):
        # memory min_decisions=3, domain min for devops=5 → effective=5
        memory = EscalationMemory(confidence_threshold=0.85, min_decisions=3)
        registry = PolicyRuleEngine()
        learner = PolicyLearner(memory=memory, registry=registry, promote_threshold=0.9)
        ctx = "hash-devops"
        # Feed 4 approvals — above memory min (3) but below domain min (5)
        for _ in range(4):
            memory.record(_make_record("devops", "approve"))
        rule = learner.promote_to_rule(ctx)
        assert rule is None, "effective_min = max(3,5) = 5; 4 decisions not enough"

    def test_promotion_not_repeated(self):
        learner, ctx = _learner_with_domain("devops", 10)
        rule1 = learner.promote_to_rule(ctx)
        rule2 = learner.promote_to_rule(ctx)
        assert rule1 is not None
        assert rule2 is None, "second promotion call must return None"
