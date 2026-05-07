from __future__ import annotations
from manifold.policy import ManifoldPolicy, PolicyDomain

def get_policy() -> ManifoldPolicy:
    domain = PolicyDomain(
        name="devops",
        escalation_threshold=0.55,
        refusal_threshold=0.40,
        verification_cost=0.10,
        penalty_scale=1.4,
        allowed_actions=["answer", "clarify", "verify", "use_tool", "retrieve", "plan", "escalate", "refuse", "wait"],
        stakes=0.6,
        risk_tolerance=0.5,
        fallback_strategy="hitl",
    )
    return ManifoldPolicy(domains=[domain])
