"""tests/test_escalation_memory.py — 6 tests for EscalationMemory + PolicyLearner."""
from __future__ import annotations

import time

import pytest

from manifold.escalation_memory import EscalationMemory, EscalationRecord
from manifold.policy_learner import PolicyLearner
from manifold.policy_rules import PolicyRuleEngine


def _make_record(decision: str, escalation_id: str = "esc-1") -> EscalationRecord:
    mem = EscalationMemory()
    context_hash = mem.make_context_hash("agent-llm", "finance", "execute_trade")
    return EscalationRecord(
        escalation_id=escalation_id,
        agent_id="agent-llm",
        action="execute_trade",
        domain="finance",
        risk_score=0.5,
        context_hash=context_hash,
        human_decision=decision,
        timestamp=time.time(),
    )


# 1. record() stores and retrieves by context_hash
def test_record_stores_and_retrieves():
    mem = EscalationMemory()
    rec = _make_record("approve", "esc-1")
    mem.record(rec)
    bucket = mem._index.get(rec.context_hash, [])
    assert len(bucket) == 1
    assert bucket[0].escalation_id == "esc-1"


# 2. should_auto_decide returns False with < 3 decisions
def test_should_auto_decide_false_with_less_than_min():
    mem = EscalationMemory(min_decisions=3)
    for i in range(2):
        mem.record(_make_record("approve", f"esc-{i}"))
    auto, decision, _ = mem.should_auto_decide("agent-llm", "execute_trade", "finance", 0.5)
    assert auto is False
    assert decision == ""


# 3. should_auto_decide returns (True, "approve") after 3 approvals
def test_should_auto_decide_approve_after_three():
    mem = EscalationMemory(min_decisions=3, confidence_threshold=0.85)
    for i in range(3):
        mem.record(_make_record("approve", f"esc-{i}"))
    auto, decision, confidence = mem.should_auto_decide("agent-llm", "execute_trade", "finance", 0.5)
    assert auto is True
    assert decision == "approve"
    assert confidence >= 0.85


# 4. should_auto_decide returns (True, "deny") after 3 denials
def test_should_auto_decide_deny_after_three():
    mem = EscalationMemory(min_decisions=3, confidence_threshold=0.85)
    for i in range(3):
        mem.record(_make_record("deny", f"esc-{i}"))
    auto, decision, confidence = mem.should_auto_decide("agent-llm", "execute_trade", "finance", 0.5)
    assert auto is True
    assert decision == "deny"
    assert confidence >= 0.85


# 5. promote_to_rule creates a PolicyRule with correct domain/action
def test_promote_to_rule_creates_policy_rule():
    mem = EscalationMemory(min_decisions=3, confidence_threshold=0.85)
    for i in range(3):
        mem.record(_make_record("approve", f"esc-{i}"))
    engine = PolicyRuleEngine()
    learner = PolicyLearner(mem, engine, promote_threshold=0.9)
    rec = list(mem._index.values())[0][0]
    rule = learner.promote_to_rule(rec.context_hash)
    assert rule is not None
    assert rule.conditions.get("domain") == "finance"
    assert rule.action in ("allow", "refuse")


# 6. weekly_summary counts auto_decided correctly
def test_weekly_summary_counts_auto_decided():
    mem = EscalationMemory()
    context_hash = mem.make_context_hash("agent-llm", "finance", "execute_trade")
    # 2 manual + 1 auto
    for i in range(2):
        rec = _make_record("approve", f"man-{i}")
        rec.auto_decided = False
        mem.record(rec)
    auto_rec = _make_record("approve", "auto-1")
    auto_rec.auto_decided = True
    mem.record(auto_rec)

    summary = mem.weekly_summary()
    assert summary["total_escalations"] == 3
    assert summary["auto_decided"] == 1
    assert summary["manual_decisions"] == 2
    assert summary["decisions_saved"] == 1
