"""manifold/policy_rules.py — Policy rule engine for MANIFOLD."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyRule:
    """A single if/then governance rule evaluated before every brain decision."""

    rule_id: str
    org_id: str
    name: str
    conditions: dict[str, Any]
    action: str  # allow / refuse / escalate / audit / etc.
    priority: int = 5
    enabled: bool = True

    def matches(self, context: dict[str, Any]) -> bool:
        """Return True if all conditions match the given context dict."""
        for key, value in self.conditions.items():
            if key == "domain":
                if context.get("domain") != value:
                    return False
            elif key == "domain_in":
                if context.get("domain") not in value:
                    return False
            elif key == "stakes_gt":
                stakes = context.get("stakes")
                if stakes is None or float(stakes) <= float(value):
                    return False
            elif key == "stakes_lt":
                stakes = context.get("stakes")
                if stakes is None or float(stakes) >= float(value):
                    return False
            elif key == "risk_gt":
                risk = context.get("risk_score")
                if risk is None or float(risk) <= float(value):
                    return False
            elif key == "org_id":
                if context.get("org_id") != value:
                    return False
            elif key == "prompt_contains":
                prompt = str(context.get("prompt", "")).lower()
                if str(value).lower() not in prompt:
                    return False
        return True

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "org_id": self.org_id,
            "name": self.name,
            "conditions": self.conditions,
            "action": self.action,
            "priority": self.priority,
            "enabled": self.enabled,
        }


class PolicyRuleEngine:
    """Evaluates ordered PolicyRules against a decision context."""

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    def evaluate(self, context: dict[str, Any]) -> str | None:
        """Return the action string of the highest-priority matching rule, or None."""
        for rule in self._rules:
            if rule.enabled and rule.matches(context):
                return rule.action
        return None

    def all_rules(self) -> list[PolicyRule]:
        return list(self._rules)

    def clear(self) -> None:
        self._rules.clear()

    def rules_for_org(self, org_id: str) -> list[PolicyRule]:
        return [r for r in self._rules if r.org_id == org_id and r.enabled]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self._rules], f)

    @classmethod
    def load(cls, path: str) -> "PolicyRuleEngine":
        engine = cls()
        if not os.path.exists(path):
            return engine
        try:
            with open(path) as f:
                data = json.load(f)
            for d in data:
                engine._rules.append(PolicyRule(**d))
            engine._rules.sort(key=lambda r: r.priority, reverse=True)
        except Exception:
            pass
        return engine
