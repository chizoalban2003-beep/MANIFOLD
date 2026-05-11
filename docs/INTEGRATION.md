# MANIFOLD Integration Guide

## Consumer quickstart — no code required

1. Open `https://your-manifold-host/signup` in a browser
2. Enter your email and organisation name
3. Copy your API key (shown once)
4. Go to `https://your-manifold-host/connect` for integration instructions
5. Open `https://your-manifold-host/report` to see your governance dashboard

That's it. Every AI call from your configured tools is now governed.

---

This guide shows how to connect your AI agent to MANIFOLD in three patterns.

---

## Universal gateway — zero code changes

The fastest integration. Change one environment variable in your
existing agent. MANIFOLD governs every call automatically.

### How it works

Instead of calling OpenAI directly, your agent calls MANIFOLD.
MANIFOLD governs the request, then forwards it to the real LLM.
Your agent receives a standard OpenAI response with a `_manifold`
metadata field appended.

### Setup (30 seconds)

Step 1 — Run MANIFOLD:
```bash
docker run -d -p 8080:8080 \
  -e MANIFOLD_API_KEY=your-key \
  -e MANIFOLD_UPSTREAM_URL=https://api.openai.com/v1 \
  -e MANIFOLD_UPSTREAM_KEY=sk-your-openai-key \
  manifold-ai
```

Step 2 — Point your agent at MANIFOLD:
```python
# Python / OpenAI SDK
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-manifold-key",
)
# That's it. Every call is now governed.
```

Step 3 — Verify governance is active:
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "test"}]
)
print(response._manifold)
# {"governed": true, "vetoed": false, "action": "answer", "risk_score": 0.08}
```

### Works with any framework

LangChain:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="your-manifold-key")
```

LlamaIndex, AutoGen, CrewAI, or any OpenAI-compatible client:
Set `base_url="http://localhost:8080/v1"` — done.

### What governance-only mode means

If `MANIFOLD_UPSTREAM_URL` is not set, MANIFOLD governs the request
but returns a governance report instead of an LLM response.
Useful for auditing without live LLM costs.

### The _manifold response field

Every response includes:

| Field | Description |
|---|---|
| `_manifold.governed` | always `true` — this call was governed |
| `_manifold.vetoed` | `true` if MANIFOLD refused the request |
| `_manifold.action` | the governance action taken |
| `_manifold.risk_score` | 0.0–1.0 risk score for this request |
| `_manifold.domain` | auto-detected domain (finance, healthcare, etc.) |

---

## Pattern 1 — Wrap any Python function with `@shield`

The `@shield` decorator intercepts any callable before it runs and runs a
MANIFOLD risk check. If risk is low the function proceeds normally. If risk is
high, a `HITLGate` is raised instead of executing the wrapped function.

```python
from manifold.brain import ManifoldBrain, BrainConfig
from manifold.interceptor import shield, InterceptorVeto

brain = ManifoldBrain(config=BrainConfig(), tools=[])

@shield(brain=brain, domain="finance", stakes=0.85)
def process_payment(amount: float) -> dict:
    # Only called when MANIFOLD approves the risk profile
    return payment_api.charge(amount)

try:
    result = process_payment(5000.0)
    print("Payment approved:", result)
except InterceptorVeto as e:
    # Risk was too high — MANIFOLD blocked the call
    print("Blocked by MANIFOLD:", e)
    notify_human_reviewer(e)
```

When `risk × stakes < threshold`: the function executes and returns its result.
When `risk × stakes ≥ threshold`: `InterceptorVeto` is raised and the function
body never runs. Catch it to route the task to your HITL queue.

---

## Pattern 2 — Call `POST /run` from any language

### Python

```python
import requests

MANIFOLD_URL = "https://your-domain"
API_KEY = "your-manifold-api-key"

response = requests.post(
    f"{MANIFOLD_URL}/run",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"prompt": "Approve vendor invoice #4821", "stakes": 0.7, "domain": "finance"},
)
result = response.json()

if result["action"] == "escalate":
    queue_for_human_review(result)
elif result["action"] == "refuse":
    raise ValueError(f"MANIFOLD refused: risk={result['risk_score']}")
else:
    proceed_with_task(result)
```

### TypeScript (manifold-ts SDK)

```typescript
import { ManifoldClient } from "manifold-ts";

const client = new ManifoldClient({
  baseUrl: "https://your-domain",
  apiKey: process.env.MANIFOLD_API_KEY,
});

const result = await client.run({
  prompt: "Approve vendor invoice #4821",
  stakes: 0.7,
  domain: "finance",
});

// result shape: { action, domain, risk_score, nearest_cells, flagged_tools }
if (result.action === "escalate") {
  await notifyOperator(result);
} else {
  await proceedWithTask(result);
}
```

**Response fields:**
- `action` — MANIFOLD's decision: `answer`, `escalate`, `refuse`, `use_tool`, etc.
- `risk_score` — float 0–1; higher means more dangerous.
- `domain` — the domain MANIFOLD routed to.
- `flagged_tools` — tools that triggered co-occurrence anomaly warnings.

---

## Pattern 3 — Shadow mode (observe before committing)

Shadow mode lets MANIFOLD watch your existing agent without changing any live
behaviour. After accumulating enough observations you can flip to live mode.

```python
from manifold.brain import ManifoldBrain, BrainConfig, BrainTask
from manifold.connector import ShadowModeWrapper

brain = ManifoldBrain(config=BrainConfig(), tools=[])

# active=False → observe only, never block
wrapper = ShadowModeWrapper(brain=brain, active=False)

# Feed your existing task stream
for task_data in production_tasks:
    task = BrainTask(
        prompt=task_data["prompt"],
        domain=task_data.get("domain", "general"),
        stakes=task_data.get("stakes", 0.5),
    )
    wrapper.observe(task, actual_action=task_data["agent_action"])

# Read what MANIFOLD would have decided
report = wrapper.shadow_report()
print(f"Disagreement rate: {report['disagreement_rate']:.1%}")
print(f"Virtual regret:    {report['virtual_regret_total']:.2f}")

# When you're confident, go live
wrapper.activate()   # now MANIFOLD decisions override the naive agent
```

The shadow report shows where MANIFOLD disagrees with your current agent and
by how much — without ever touching production traffic.

---

## What to do when `action == "escalate"`

When MANIFOLD returns `action: "escalate"` it means the task's expected regret
exceeds the domain's escalation threshold. MANIFOLD is not refusing the task
outright — it is asking a human to review before the agent acts.

The recommended pattern is a lightweight escalation queue:

1. **Log the task.** Write the full `/run` response (including `prompt`,
   `risk_score`, `domain`, and `flagged_tools`) to a durable store such as a
   database table or a message queue. Include a timestamp and a unique
   `task_id`.

2. **Notify an operator.** Send an alert — email, Slack, PagerDuty — with a
   link to a review UI that shows the task details and the MANIFOLD risk
   summary. Keep the notification short: domain, risk score, and the prompt.

3. **Wait for human approval.** The task stays in the queue in a `pending`
   state. Your agent must not proceed until a human approves (or rejects) it.
   Use a webhook or polling loop to watch for the operator's decision.

4. **Resume or cancel.** If the operator approves, re-submit the task to your
   agent with a flag indicating human clearance. If rejected, log the outcome
   and notify the requester that the task was blocked.

Escalation is not a failure — it is MANIFOLD doing its job. Over time, as
MANIFOLD calibrates its thresholds from your domain's outcome history, the
escalation rate will converge to the target set in your domain pack.
