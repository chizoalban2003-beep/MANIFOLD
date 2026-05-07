# MANIFOLD Domain Packs Reference

---

## Section 1 — Built-in domain packs

| Name | `escalation_threshold` | `refusal_threshold` | `penalty_scale` | Key use cases |
|---|---|---|---|---|
| `healthcare` | 0.30 | 0.15 | 2.5 | Patient data, clinical decisions, medication checks |
| `finance` | 0.40 | 0.25 | 1.8 | Payments, transactions, fraud detection |
| `devops` | 0.55 | 0.40 | 1.4 | Code deployment, CI/CD pipelines, infrastructure changes |
| `legal` | 0.35 | 0.20 | 2.0 | Contract review, compliance checks, regulatory filings |
| `infrastructure` | 0.20 | 0.08 | 3.5 | Cloud provisioning, network changes, database migrations |
| `trading` | 0.45 | 0.30 | 2.5 | Algorithmic trading, market orders, portfolio rebalancing |
| `supply_chain` | 0.52 | 0.38 | 1.5 | Inventory management, logistics, vendor payments |

---

## Section 2 — How thresholds work

**`escalation_threshold`**: when a task's `expected_regret` exceeds this value,
MANIFOLD routes to a human reviewer rather than acting autonomously. Lower values
mean earlier escalation (more conservative).

**`refusal_threshold`**: when risk exceeds this value, MANIFOLD refuses the task
entirely and raises `InterceptorVeto`. This is a hard stop — no human review,
no fallback.

**`penalty_scale`**: multiplier applied to the base penalty for wrong autonomous
actions. A `penalty_scale` of 2.5 means a bad decision in this domain is 2.5×
more costly than in a neutral domain, steering the policy toward caution.

---

## Section 3 — How to choose a domain

If your tasks involve patient data, clinical notes, or medical decisions → use
**healthcare**. If your tasks involve financial transactions, payments, or fraud
signals → use **finance**. If your tasks involve code deployment, CI pipelines,
or infrastructure changes → use **devops**. When uncertain → use **general**
(lowest thresholds, most permissive, suitable for development and low-stakes
automation).

---

## Section 4 — How to create a custom domain pack

### Step 1 — Create the domain module

**File: `manifold/domains/ecommerce_returns.py`**

```python
from __future__ import annotations
from manifold.policy import ManifoldPolicy, PolicyDomain

def get_policy() -> ManifoldPolicy:
    domain = PolicyDomain(
        name="ecommerce_returns",
        escalation_threshold=0.55,
        refusal_threshold=0.40,
        verification_cost=0.08,
        penalty_scale=1.3,
        allowed_actions=[
            "answer", "clarify", "verify", "use_tool",
            "retrieve", "wait", "escalate", "refuse",
        ],
        stakes=0.45,
        risk_tolerance=0.50,
        fallback_strategy="hitl",
        notes="E-commerce returns and refunds; moderate risk, fast resolution.",
    )
    return ManifoldPolicy(domains=[domain])
```

### Step 2 — Register in `manifold/domains/__init__.py`

```python
_DOMAIN_MODULES = {
    # ... existing domains ...
    "ecommerce_returns": "manifold.domains.ecommerce_returns",
}
```

### Step 3 — Use it

```python
from manifold.domains import load_domain

policy = load_domain("ecommerce_returns")
print(policy.domains[0].escalation_threshold)  # 0.55
```

---

## Section 5 — Auto-routing vs explicit domain

Set `domain` explicitly when you know the business context upfront (e.g. all
tasks in your payment service should use `finance`). Let `GlobalWorkspace`
route automatically when your agent handles mixed workloads across multiple
domains — it will keyword-match the prompt and competition-score candidate
domains to pick the best fit.
