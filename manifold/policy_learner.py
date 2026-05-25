"""manifold/policy_learner.py — PolicyLearner.

Scans EscalationMemory for high-confidence patterns and promotes them
to formal PolicyRule entries so MANIFOLD never has to ask the same
question again.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.escalation_memory import EscalationMemory
    from manifold.policy_rules import PolicyRule, PolicyRuleEngine


class PolicyLearner:
    """Promotes repeated escalation patterns to automatic rules.

    Parameters
    ----------
    memory:
        An :class:`~manifold.escalation_memory.EscalationMemory` instance.
    registry:
        A :class:`~manifold.policy_rules.PolicyRuleEngine` instance where
        promoted rules are registered.
    promote_threshold:
        Minimum approval-rate confidence to promote a rule.  Defaults to
        0.9 (stricter than the 0.85 auto-decide threshold).
    """

    def __init__(
        self,
        memory: "EscalationMemory",
        registry: "PolicyRuleEngine",
        promote_threshold: float = 0.9,
    ) -> None:
        self._memory = memory
        self._registry = registry
        self._promote_threshold = promote_threshold
        # Track which context hashes we have already promoted
        self._promoted: set[str] = set()

    # ------------------------------------------------------------------

    def promote_to_rule(self, context_hash: str) -> "PolicyRule | None":
        """Attempt to promote a context_hash pattern to a PolicyRule.

        Returns the new :class:`~manifold.policy_rules.PolicyRule` if
        promoted, or ``None`` if the pattern does not meet the threshold.
        """
        from manifold.policy_rules import PolicyRule  # noqa: PLC0415

        if context_hash in self._promoted:
            return None

        bucket = self._memory._index.get(context_hash, [])
        if not bucket:
            return None

        n = len(bucket)
        if n < self._memory._min:
            return None

        approvals = sum(1 for r in bucket if r.human_decision == "approve")
        approval_rate = approvals / n

        # Use the most recent record for domain / action / risk metadata
        sample = bucket[-1]
        domain = sample.domain
        agent_type = sample.agent_id  # stored verbatim; use as label
        action = sample.action
        risk_score = sample.risk_score

        if approval_rate >= self._promote_threshold:
            effect = "allow"
            rule = PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=getattr(sample, "org_id", "default"),
                name=f"auto_approve:{domain}:{action[:30]}",
                conditions={
                    "domain": domain,
                    "risk_gt": 0.0,
                },
                action=effect,
                priority=5,
            )
            # Store the stakes ceiling based on seen risk
            rule.conditions["risk_gt"] = 0.0
            # Upper bound: only apply up to risk_score * 1.1
            rule.conditions["stakes_lt"] = min(risk_score * 1.1, 1.0) + 0.01
            self._registry.add_rule(rule)
            self._promoted.add(context_hash)
            print(f"MANIFOLD learned: {rule.name} from {n} decisions")
            return rule

        if approval_rate <= 1.0 - self._promote_threshold:
            rule = PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=getattr(sample, "org_id", "default"),
                name=f"auto_deny:{domain}:{action[:30]}",
                conditions={
                    "domain": domain,
                },
                action="refuse",
                priority=5,
                enabled=True,
            )
            self._registry.add_rule(rule)
            self._promoted.add(context_hash)
            print(
                f"MANIFOLD learned: {rule.name} (auto_deny) from {n} decisions"
                f" — agent_type={agent_type}"
            )
            return rule

        return None

    def scan_and_promote(self) -> "list[PolicyRule]":
        """Scan all context hashes and promote qualifying patterns.

        Returns the list of newly promoted :class:`~manifold.policy_rules.PolicyRule`
        objects (empty list if nothing qualifies).
        """
        promoted = []
        for context_hash in list(self._memory._index):
            rule = self.promote_to_rule(context_hash)
            if rule is not None:
                promoted.append(rule)
        return promoted
