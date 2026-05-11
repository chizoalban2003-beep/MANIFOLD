"""tests/test_policy_translator.py — Tests for PolicyTranslator."""
from __future__ import annotations

import pytest

from manifold.policy_translator import (
    PolicyTranslator,
    translate_domain,
    VALID_ACTIONS,
    VALID_CONDITIONS,
)
from manifold.policy_rules import PolicyRule


def test_valid_rule_returns_policy_rule():
    """A fully valid dict should produce a PolicyRule."""
    pt = PolicyTranslator(org_id="test-org")
    data = {
        "name": "Block PHI",
        "conditions": {"domain": "healthcare", "risk_gt": 0.7},
        "action": "refuse",
        "priority": 85,
    }
    rule = pt.validate_rule(data)
    assert isinstance(rule, PolicyRule)
    assert rule.name == "Block PHI"
    assert rule.action == "refuse"
    assert rule.priority == 85
    assert rule.conditions["domain"] == "healthcare"


def test_invalid_action_raises():
    """Unknown action value should raise ValueError with a plain English message."""
    pt = PolicyTranslator(org_id="test-org")
    with pytest.raises(ValueError, match="not a valid MANIFOLD action"):
        pt.validate_rule({"name": "Bad rule", "action": "destroy", "priority": 50})


def test_invalid_condition_key_raises():
    """Unknown condition key should raise ValueError."""
    pt = PolicyTranslator(org_id="test-org")
    with pytest.raises(ValueError, match="Unknown condition key"):
        pt.validate_rule({"name": "Bad cond", "conditions": {"foo_bar": "baz"}, "action": "allow"})


def test_domain_vocabulary_mapping():
    """Natural language terms should map to MANIFOLD domain strings."""
    assert translate_domain("patient") == "healthcare"
    assert translate_domain("payroll") == "finance"
    assert translate_domain("contract") == "legal"
    assert translate_domain("deploy") == "devops"
    assert translate_domain("unknown_term") == "unknown_term"


def test_hipaa_preset_returns_policy_rules():
    """HIPAA preset should return at least 4 PolicyRule objects."""
    rules = PolicyTranslator.hipaa_preset(org_id="test-org")
    assert len(rules) >= 4
    assert all(isinstance(r, PolicyRule) for r in rules)
    actions = {r.action for r in rules}
    assert actions & {"refuse", "audit", "escalate", "require_approval", "redact"}


def test_gdpr_preset_contains_erasure_rule():
    """GDPR preset should include a right-to-erasure rule."""
    rules = PolicyTranslator.gdpr_preset(org_id="test-org")
    names = " ".join(r.name.lower() for r in rules)
    assert "erasure" in names or "delete" in names or "gdpr art.17" in names.lower()


def test_apply_preset_sox():
    """apply_preset('sox') should return SOX rules."""
    rules = PolicyTranslator.apply_preset("sox", org_id="test-org")
    assert len(rules) >= 4
    assert all(r.org_id == "test-org" for r in rules)


def test_apply_preset_unknown_raises():
    """apply_preset with unknown name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown preset"):
        PolicyTranslator.apply_preset("foobar")
