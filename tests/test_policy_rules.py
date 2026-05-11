"""Tests for manifold/policy_rules.py — PolicyRule and PolicyRuleEngine."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from manifold.policy_rules import PolicyRule, PolicyRuleEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rule(
    *,
    org_id: str = "org1",
    conditions: dict | None = None,
    action: str = "refuse",
    priority: int = 0,
    enabled: bool = True,
) -> PolicyRule:
    return PolicyRule(
        rule_id=str(uuid.uuid4()),
        org_id=org_id,
        name="test rule",
        conditions=conditions or {},
        action=action,
        priority=priority,
        enabled=enabled,
    )


def _ctx(**kwargs) -> dict:
    base = {
        "domain": "finance",
        "stakes": 0.5,
        "risk_score": 0.3,
        "prompt": "do something",
        "org_id": "org1",
        "tools_used": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# PolicyRule.matches() tests
# ---------------------------------------------------------------------------


def test_matches_domain_condition() -> None:
    """Rule with domain= condition matches only the correct domain."""
    rule = _rule(conditions={"domain": "finance"})
    assert rule.matches(_ctx(domain="finance")) is True
    assert rule.matches(_ctx(domain="healthcare")) is False


def test_matches_domain_in_condition() -> None:
    """Rule with domain_in= matches any domain in the list."""
    rule = _rule(conditions={"domain_in": ["finance", "legal"]})
    assert rule.matches(_ctx(domain="finance")) is True
    assert rule.matches(_ctx(domain="legal")) is True
    assert rule.matches(_ctx(domain="devops")) is False


def test_matches_stakes_gt_condition() -> None:
    """Rule with stakes_gt= matches when stakes exceed threshold."""
    rule = _rule(conditions={"stakes_gt": 0.7})
    assert rule.matches(_ctx(stakes=0.8)) is True
    assert rule.matches(_ctx(stakes=0.7)) is False  # strictly greater
    assert rule.matches(_ctx(stakes=0.5)) is False


def test_matches_prompt_contains_case_insensitive() -> None:
    """Rule with prompt_contains= is case-insensitive."""
    rule = _rule(conditions={"prompt_contains": "delete"})
    assert rule.matches(_ctx(prompt="Please DELETE all records")) is True
    assert rule.matches(_ctx(prompt="delete this")) is True
    assert rule.matches(_ctx(prompt="save records")) is False


def test_matches_returns_false_when_condition_not_met() -> None:
    """Rule with multiple conditions fails when any one is not met."""
    rule = _rule(conditions={"domain": "finance", "risk_gt": 0.8})
    # risk_score is 0.3 → doesn't satisfy risk_gt=0.8
    assert rule.matches(_ctx(domain="finance", risk_score=0.3)) is False
    # both satisfied
    assert rule.matches(_ctx(domain="finance", risk_score=0.9)) is True


def test_to_dict_returns_serialisable_dict() -> None:
    """to_dict() returns a dict that is JSON-serialisable."""
    rule = _rule(conditions={"domain": "finance"})
    d = rule.to_dict()
    assert isinstance(d, dict)
    json_str = json.dumps(d)  # must not raise
    assert "rule_id" in d
    assert "conditions" in d
    assert d["action"] == "refuse"


# ---------------------------------------------------------------------------
# PolicyRuleEngine tests
# ---------------------------------------------------------------------------


def test_evaluate_returns_action_of_matching_rule() -> None:
    """evaluate() returns the action of the matching rule."""
    engine = PolicyRuleEngine()
    engine.add_rule(_rule(conditions={"domain": "finance"}, action="refuse"))
    action = engine.evaluate(_ctx(domain="finance"))
    assert action == "refuse"


def test_higher_priority_rule_wins() -> None:
    """Higher priority rule is evaluated first and its action wins."""
    engine = PolicyRuleEngine()
    engine.add_rule(_rule(conditions={"domain": "finance"}, action="allow", priority=1))
    engine.add_rule(_rule(conditions={"domain": "finance"}, action="refuse", priority=10))
    # priority=10 should win
    action = engine.evaluate(_ctx(domain="finance"))
    assert action == "refuse"


def test_evaluate_returns_none_when_nothing_matches() -> None:
    """evaluate() returns None when no rule matches."""
    engine = PolicyRuleEngine()
    engine.add_rule(_rule(conditions={"domain": "healthcare"}, action="refuse"))
    action = engine.evaluate(_ctx(domain="finance"))
    assert action is None


def test_save_load_round_trip_preserves_rules(tmp_path: Path) -> None:
    """save() then load() restores all rules with correct fields."""
    engine = PolicyRuleEngine()
    engine.add_rule(_rule(conditions={"domain": "finance"}, action="refuse", priority=5))
    engine.add_rule(_rule(conditions={"stakes_gt": 0.9}, action="escalate", priority=1))

    path = str(tmp_path / "rules.json")
    engine.save(path)

    restored = PolicyRuleEngine.load(path)
    assert len(restored.all_rules()) == 2
    # Highest priority first
    assert restored.all_rules()[0].priority == 5
    assert restored.all_rules()[0].action == "refuse"
    assert restored.all_rules()[1].action == "escalate"


def test_load_nonexistent_returns_fresh_engine() -> None:
    """load() on a missing path returns a fresh empty engine without error."""
    engine = PolicyRuleEngine.load("/tmp/manifold_nonexistent_rules_xyz.json")
    assert isinstance(engine, PolicyRuleEngine)
    assert engine.all_rules() == []


def test_remove_rule_returns_true_and_deletes() -> None:
    """remove_rule() returns True and removes the rule."""
    engine = PolicyRuleEngine()
    r = _rule(conditions={"domain": "legal"})
    engine.add_rule(r)
    assert len(engine.all_rules()) == 1
    result = engine.remove_rule(r.rule_id)
    assert result is True
    assert len(engine.all_rules()) == 0


def test_remove_rule_returns_false_for_unknown_id() -> None:
    """remove_rule() returns False for an unknown rule id."""
    engine = PolicyRuleEngine()
    result = engine.remove_rule("nonexistent-id")
    assert result is False


def test_rules_for_org_returns_only_enabled_rules_for_org() -> None:
    """rules_for_org() returns only enabled rules for the specified org."""
    engine = PolicyRuleEngine()
    engine.add_rule(_rule(org_id="org1", action="allow", enabled=True))
    engine.add_rule(_rule(org_id="org1", action="refuse", enabled=False))
    engine.add_rule(_rule(org_id="org2", action="allow", enabled=True))
    rules = engine.rules_for_org("org1")
    assert len(rules) == 1
    assert rules[0].action == "allow"
