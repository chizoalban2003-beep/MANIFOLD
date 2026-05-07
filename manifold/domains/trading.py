from __future__ import annotations
from manifold.policy import ManifoldPolicy, PolicyDomain

def get_policy() -> ManifoldPolicy:
    domain = PolicyDomain(
        name="trading",
        escalation_threshold=0.45,
        refusal_threshold=0.30,
        verification_cost=0.12,
        penalty_scale=2.5,
        allowed_actions=["answer", "verify", "use_tool", "wait", "escalate", "refuse", "stop"],
        stakes=0.85,
        risk_tolerance=0.3,
        fallback_strategy="hitl",
    )
    return ManifoldPolicy(domains=[domain])
