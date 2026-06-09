"""EXP-C — PolicyLearner Convergence Rate by Domain.

Instruments PolicyLearner to measure how many escalations occur before
and after policy promotion, comparing domains to empirically validate
the per-domain minimum decision threshold recommendation.

No new dependencies.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any

from manifold.escalation_memory import EscalationMemory, EscalationRecord
from manifold.policy_learner import PolicyLearner
from manifold.policy_rules import PolicyRuleEngine


@dataclass
class PromotionEvent:
    domain: str
    n_decisions: int
    approval_rate: float
    rule_type: str  # "auto_approve" or "auto_deny"
    escalations_before: int
    escalations_after_sim: int  # simulated: if rule fires, escalations drop


def _simulate_domain_escalations(
    domain: str,
    approval_rate: float,
    n_decisions: int,
    n_post_decisions: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Simulate escalation lifecycle for a single domain pattern.

    Returns dict with promotion details and escalation savings estimate.
    """
    memory = EscalationMemory(confidence_threshold=0.85, min_decisions=3)
    registry = PolicyRuleEngine()

    # Use domain-specific min for the learner
    learner = PolicyLearner(
        memory=memory,
        registry=registry,
        promote_threshold=0.9,
    )

    agent_id = f"{domain}_agent"
    action = "execute_transaction"
    base_hash = EscalationMemory.make_context_hash(agent_id, domain, action)
    effective_min = learner._min_for_domain(domain)

    promoted_at = None
    escalations_before_promotion = 0

    # Feed decisions until promotion triggers
    for i in range(n_decisions):
        decision = "approve" if rng.random() < approval_rate else "deny"
        rec = EscalationRecord(
            escalation_id=str(uuid.uuid4()),
            agent_id=agent_id,
            action=action,
            domain=domain,
            risk_score=0.65,
            context_hash=base_hash,
            human_decision=decision,
        )
        memory.record(rec)
        escalations_before_promotion += 1

        # Try promotion after each decision
        rule = learner.promote_to_rule(base_hash)
        if rule is not None:
            promoted_at = i + 1
            break

    # Simulate post-promotion: escalations that would have happened
    # but were auto-decided instead (saved escalations)
    escalations_after = 0
    if promoted_at is not None:
        # Any remaining decisions that exceed effective_min are now auto-decided
        remaining = max(0, n_decisions + n_post_decisions - promoted_at)
        # Each one would have been an escalation without the rule
        escalations_after = 0  # zero: the rule handles them automatically
        saved = remaining
    else:
        saved = 0
        escalations_after = n_post_decisions  # no promotion, all still escalate

    return {
        "domain": domain,
        "effective_min": effective_min,
        "promoted": promoted_at is not None,
        "promoted_at_decision": promoted_at,
        "escalations_before": escalations_before_promotion,
        "escalations_saved": saved,
        "total_decisions": n_decisions + (n_post_decisions if promoted_at else 0),
        "rules_created": len(registry.all_rules()),
    }


def run_policy_learner_convergence_benchmark() -> dict[str, Any]:
    """Simulate PolicyLearner across all domains and measure convergence.

    For each domain:
    - Feed decisions with 92% approval rate (triggers auto_approve promotion)
    - Record how many escalations occur before promotion fires
    - Simulate 100 post-promotion decisions to quantify savings

    Returns domain-by-domain comparison + summary statistics.
    """
    rng = random.Random(2025)
    approval_rate = 0.92
    n_decisions = 200   # upper bound; promotion may fire earlier
    n_post = 100        # decisions simulated after promotion

    domains = [
        "devops",
        "infrastructure",
        "supply_chain",
        "trading",
        "finance",
        "general",
        "legal",
        "healthcare",
    ]

    domain_results = []
    for domain in domains:
        result = _simulate_domain_escalations(
            domain, approval_rate, n_decisions, n_post, rng
        )
        domain_results.append(result)

    # Summary
    promoted_domains = [r for r in domain_results if r["promoted"]]
    avg_decisions_to_promote = (
        sum(r["promoted_at_decision"] for r in promoted_domains) / len(promoted_domains)
        if promoted_domains
        else 0.0
    )
    total_escalations_before = sum(r["escalations_before"] for r in domain_results)
    total_saved = sum(r["escalations_saved"] for r in domain_results)

    return {
        "domain_results": domain_results,
        "domains_promoted": len(promoted_domains),
        "domains_total": len(domains),
        "avg_decisions_to_promote": round(avg_decisions_to_promote, 1),
        "total_escalations_before_promotion": total_escalations_before,
        "total_escalations_saved": total_saved,
        "savings_ratio": round(
            total_saved / max(total_escalations_before + total_saved, 1), 4
        ),
    }
