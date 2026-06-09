"""manifold/policy_rules.py — Policy rule engine for MANIFOLD."""

from __future__ import annotations

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
            ctx_val = context.get(key)
            if key == "domain":
                if ctx_val != value:
                    return False
            elif key == "domain_in":
                if ctx_val not in value:
                    return False
            elif key == "stakes_gt":
                if ctx_val is None or float(ctx_val) <= float(value):
                    return False
            elif key == "stakes_lt":
                if ctx_val is None or float(ctx_val) >= float(value):
                    return False
            elif key == "risk_gt":
                if ctx_val is None or float(ctx_val) <= float(value):
                    return False
            elif key == "org_id":
                if ctx_val != value:
                    return False
        return True


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

    def evaluate(self, context: dict[str, Any]) -> PolicyRule | None:
        """Return the highest-priority matching enabled rule, or None."""
        for rule in self._rules:
            if rule.enabled and rule.matches(context):
                return rule
        return None

    def all_rules(self) -> list[PolicyRule]:
        return list(self._rules)

    def clear(self) -> None:
        self._rules.clear()
