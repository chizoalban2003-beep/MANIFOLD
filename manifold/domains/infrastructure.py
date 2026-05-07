from __future__ import annotations
from manifold.policy import ManifoldPolicy, PolicyDomain

def get_policy() -> ManifoldPolicy:
    domain = PolicyDomain(
        name="infrastructure",
        escalation_threshold=0.20,
        refusal_threshold=0.08,
        verification_cost=0.30,
        penalty_scale=3.5,
        allowed_actions=["verify", "clarify", "retrieve", "wait", "escalate", "refuse"],
        stakes=0.85,
        risk_tolerance=0.15,
        fallback_strategy="hitl",
    )
    return ManifoldPolicy(domains=[domain])
