from __future__ import annotations
from manifold.policy import ManifoldPolicy, PolicyDomain

def get_policy() -> ManifoldPolicy:
    domain = PolicyDomain(
        name="supply_chain",
        escalation_threshold=0.52,
        refusal_threshold=0.38,
        verification_cost=0.08,
        penalty_scale=1.5,
        allowed_actions=["answer", "clarify", "verify", "use_tool", "retrieve", "wait", "escalate", "refuse"],
        stakes=0.55,
        risk_tolerance=0.45,
        fallback_strategy="hitl",
    )
    return ManifoldPolicy(domains=[domain])
