"""manifold/policy_learner.py — PolicyLearner.

Scans EscalationMemory for high-confidence patterns and promotes them
to formal PolicyRule entries so MANIFOLD never has to ask the same
question again.

Fix (8.5): Per-domain minimum decision thresholds. Low-frequency
high-stakes domains (healthcare, legal) require significantly more
confirmed decisions before auto-promotion than high-frequency domains
(devops, finance).  The global confidence threshold is unchanged; only
the minimum sample count is domain-specific.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.escalation_memory import EscalationMemory
    from manifold.policy_rules import PolicyRule, PolicyRuleEngine


# Minimum number of consistent decisions required before promotion, keyed by
# domain.  Calibrated to decision frequency and consequence severity.
# Domains absent from this dict fall back to the instance default_min_decisions.
DOMAIN_MIN_DECISIONS: dict[str, int] = {
    # High-frequency, easily recoverable
    "devops": 5,
    "infrastructure": 5,
    "supply_chain": 8,
    "trading": 10,
    # Medium-frequency, moderate stakes
    "finance": 15,
    "general": 10,
    # Low-frequency, hard to reverse — high bar before trusting the pattern
    "legal": 50,
    "healthcare": 50,
}


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
    default_min_decisions:
        Fallback minimum decision count for domains not found in
        ``DOMAIN_MIN_DECISIONS``.  Defaults to 10.
    domain_min_decisions:
        Optional caller-supplied overrides merged on top of the module-level
        ``DOMAIN_MIN_DECISIONS`` defaults, keyed by domain string.
    """

    def __init__(
        self,
        memory: "EscalationMemory",
        registry: "PolicyRuleEngine",
        promote_threshold: float = 0.9,
        default_min_decisions: int = 10,
        domain_min_decisions: dict[str, int] | None = None,
    ) -> None:
        self._memory = memory
        self._registry = registry
        self._promote_threshold = promote_threshold
        self._default_min = default_min_decisions
        self._domain_min: dict[str, int] = dict(DOMAIN_MIN_DECISIONS)
        if domain_min_decisions:
            self._domain_min.update(domain_min_decisions)
        self._promoted: set[str] = set()

    # ------------------------------------------------------------------

    def _min_for_domain(self, domain: str) -> int:
        """Return the minimum decision count required for *domain*."""
        return self._domain_min.get(domain.lower(), self._default_min)

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
        sample = bucket[-1]
        domain = sample.domain

        # Domain-specific minimum takes precedence over the memory global min.
        # Use the stricter of the two so neither can be bypassed by config.
        effective_min = max(self._memory._min, self._min_for_domain(domain))

        if n < effective_min:
            return None

        approvals = sum(1 for r in bucket if r.human_decision == "approve")
        approval_rate = approvals / n

        agent_type = sample.agent_id
        action = sample.action
        risk_score = sample.risk_score

        if approval_rate >= self._promote_threshold:
            rule = PolicyRule(
                rule_id=str(uuid.uuid4()),
                org_id=getattr(sample, "org_id", "default"),
                name=f"auto_approve:{domain}:{action[:30]}",
                conditions={
                    "domain": domain,
                    "risk_gt": 0.0,
                },
                action="allow",
                priority=5,
            )
            rule.conditions["risk_gt"] = 0.0
            rule.conditions["stakes_lt"] = min(risk_score * 1.1, 1.0) + 0.01
            self._registry.add_rule(rule)
            self._promoted.add(context_hash)
            print(
                f"MANIFOLD learned: {rule.name} from {n} decisions "
                f"(domain_min={effective_min})"
            )
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
                f"MANIFOLD learned: {rule.name} (auto_deny) from {n} decisions "
                f"(domain_min={effective_min}) — agent_type={agent_type}"
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
