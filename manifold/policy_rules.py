"""manifold/policy_rules.py — Policy-as-code engine for MANIFOLD.

Allows operators to define if/then rules that short-circuit the brain
when a matching condition is found. Rules are evaluated in priority order
(highest first); the first match wins.

Example::

    from manifold.policy_rules import PolicyRule, PolicyRuleEngine
    import uuid

    engine = PolicyRuleEngine()
    engine.add_rule(PolicyRule(
        rule_id=str(uuid.uuid4()),
        org_id='org1',
        name='Block high-risk finance',
        conditions={'domain': 'finance', 'risk_gt': 0.8},
        action='refuse',
        priority=10,
    ))
    action = engine.evaluate({
        'domain': 'finance', 'stakes': 0.5, 'risk_score': 0.9,
        'prompt': '', 'org_id': 'org1', 'tools_used': [],
    })
    # action == 'refuse'
"""
from __future__ import annotations

import json
import re as _re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PolicyRule:
    """A single if/then policy rule.

    Conditions dict may contain any of:
      domain          exact domain match
      domain_in       list of domains (any match)
      stakes_gt       float — stakes must be strictly greater than
      stakes_lt       float — stakes must be strictly less than
      risk_gt         float — risk_score must be strictly greater than
      prompt_contains case-insensitive substring match on prompt
      prompt_regex    regex match on prompt
      org_id          exact org match
      tool_used       string that must appear in tools_used list
    """

    rule_id: str
    org_id: str
    name: str
    conditions: dict = field(default_factory=dict)
    action: str = "allow"
    priority: int = 0
    enabled: bool = True

    # ------------------------------------------------------------------
    def matches(self, context: dict) -> bool:
        """Return True only when ALL conditions are satisfied.

        context keys: domain, stakes, risk_score, prompt, org_id, tools_used
        """
        for key, value in self.conditions.items():
            if key == "domain":
                if context.get("domain") != value:
                    return False
            elif key == "domain_in":
                if context.get("domain") not in value:
                    return False
            elif key == "stakes_gt":
                if not (context.get("stakes", 0.0) > value):
                    return False
            elif key == "stakes_lt":
                if not (context.get("stakes", 0.0) < value):
                    return False
            elif key == "risk_gt":
                if not (context.get("risk_score", 0.0) > value):
                    return False
            elif key == "prompt_contains":
                prompt = context.get("prompt", "") or ""
                if value.lower() not in prompt.lower():
                    return False
            elif key == "prompt_regex":
                prompt = context.get("prompt", "") or ""
                if not _re.search(value, prompt):
                    return False
            elif key == "org_id":
                if context.get("org_id") != value:
                    return False
            elif key == "tool_used":
                tools = context.get("tools_used") or []
                if value not in tools:
                    return False
        return True

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)


class PolicyRuleEngine:
    """Evaluates a sorted list of PolicyRule objects against a context.

    Rules are sorted by priority descending; highest priority wins.
    """

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []

    # ------------------------------------------------------------------
    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule and re-sort by priority descending."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    # ------------------------------------------------------------------
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by id. Returns True if found and removed."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    # ------------------------------------------------------------------
    def rules_for_org(self, org_id: str) -> list[PolicyRule]:
        """Return enabled rules for the given org, in priority order."""
        return [r for r in self._rules if r.enabled and r.org_id == org_id]

    # ------------------------------------------------------------------
    def evaluate(self, context: dict) -> str | None:
        """Evaluate rules for context['org_id'] in priority order.

        Returns the action of the FIRST matching rule, or None if nothing
        matches (in which case the brain decides as normal).
        """
        org_id = context.get("org_id", "")
        for rule in self.rules_for_org(org_id):
            if rule.matches(context):
                return rule.action
        return None

    # ------------------------------------------------------------------
    def all_rules(self) -> list[PolicyRule]:
        """Return all rules (enabled and disabled)."""
        return list(self._rules)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise all rules to a JSON file at path."""
        data = [r.to_dict() for r in self._rules]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"rules": data}, fh, indent=2)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "PolicyRuleEngine":
        """Deserialise from a JSON file at path.

        Returns a fresh empty engine if the file does not exist.
        """
        engine = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for rule_dict in data.get("rules", []):
                engine.add_rule(PolicyRule(**rule_dict))
        except FileNotFoundError:
            pass
        return engine
