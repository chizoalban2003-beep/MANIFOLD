# MANIFOLD HTTP API Reference

Base URL: `https://your-domain` (or `http://localhost:8080` locally)

All JSON responses use `Content-Type: application/json`.
Protected endpoints require `Authorization: Bearer <MANIFOLD_API_KEY>`.

---

### POST /run
Execute the full MANIFOLD pipeline: encode → route → decide → flag.

**Auth:** Bearer token required

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | ✅ | Task description |
| `stakes` | float | | Risk stakes 0–1 (default 0.5) |
| `domain` | string | | Override domain routing |
| `uncertainty` | float | | Uncertainty 0–1 |
| `data` | any | | Structured payload forwarded to encoder |
| `tools_used` | list[string] | | Tool names used in this task |

**Response:**

```json
{
  "action": "escalate",
  "domain": "finance",
  "risk_score": 0.72,
  "nearest_cells": [{"row": 3, "col": 7, "distance": 0.12}],
  "flagged_tools": ["billing_api"]
}
```

**Example:**

```bash
curl -X POST https://your-domain/run \
  -H "Authorization: Bearer $MANIFOLD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Process a $50,000 wire transfer", "stakes": 0.9, "domain": "finance"}'
```

---

### POST /shield
Single-task risk decision via the ActiveInterceptor (@shield).

**Auth:** Bearer token required

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | ✅ | Task description |
| `domain` | string | | Domain context |
| `stakes` | float | | Stakes 0–1 |
| `uncertainty` | float | | Uncertainty 0–1 |
| `complexity` | float | | Complexity 0–1 |

**Response:**

```json
{
  "vetoed": true,
  "reason": "brain action='escalate'; risk=0.812",
  "risk_score": 0.8120,
  "confidence": 0.7430,
  "suggested_action": "escalate"
}
```

**Example:**

```bash
curl -X POST https://your-domain/shield \
  -H "Authorization: Bearer $MANIFOLD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Delete all user records", "stakes": 0.95}'
```

---

### GET /learned
Return a snapshot of everything the system has learned: cognitive map, tool co-occurrence patterns, and promoted consolidation rules.

**Auth:** Public

**Response:**

```json
{
  "cognitive_map": {"total_outcomes": 142, "recent_actions": ["answer", "escalate"]},
  "cooccurrence": {"total_tools": 5, "top_pairs": []},
  "consolidation": [{"domain": "finance", "action": "escalate", "confidence": 0.87}],
  "prediction": {"calibration_error": 0.04}
}
```

**Example:**

```bash
curl https://your-domain/learned
```

---

### GET /metrics
Prometheus-compatible plain-text metrics for monitoring.

**Auth:** Public

**Response:** `text/plain` in Prometheus exposition format.

```
# HELP manifold_tasks_total Total tasks processed
# TYPE manifold_tasks_total counter
manifold_tasks_total 1024
# HELP manifold_escalations_total Tasks escalated to human
manifold_escalations_total 38
# HELP manifold_refusals_total Tasks refused
manifold_refusals_total 12
```

**Example:**

```bash
curl https://your-domain/metrics
```

---

### GET /policy
Return the server's active domain policy as JSON.

**Auth:** Public

**Response:**

```json
{
  "org_id": "manifold-server",
  "risk_tolerance": 0.45,
  "coordination_tax_cap": 0.2,
  "fallback_strategy": "hitl",
  "min_tool_reliability": 0.7
}
```

**Example:**

```bash
curl https://your-domain/policy
```

---

### GET /ats/score/\<tool_id\>
Return the Agent Trust Score for a specific tool.

**Auth:** Public

**Path parameter:** `tool_id` — the registered tool identifier.

**Response:**

```json
{
  "tool_id": "billing_api",
  "composite_score": 0.732,
  "signal_count": 47,
  "domain": "finance",
  "display_name": "Billing API"
}
```

**Example:**

```bash
curl https://your-domain/ats/score/billing_api
```

---

### GET /ats/leaderboard
Return the top 10 tools by trust score.

**Auth:** Public

**Response:** JSON array of Agent Trust Score objects (same shape as `/ats/score/<id>`).

```json
[
  {"tool_id": "billing_api", "composite_score": 0.91, "signal_count": 120},
  {"tool_id": "crm_api",     "composite_score": 0.87, "signal_count": 88}
]
```

**Example:**

```bash
curl https://your-domain/ats/leaderboard
```

---

### POST /ats/register
Register a new tool in the ATS trust network.

**Auth:** Bearer token required

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_id` | string | ✅ | Unique tool identifier |
| `org_id` | string | | Owning organisation |
| `display_name` | string | | Human-readable name |
| `domain` | string | | Domain context (default `general`) |
| `description` | string | | Tool description |

**Response:**

```json
{
  "registered": true,
  "tool_id": "payment_gateway_v2"
}
```

**Example:**

```bash
curl -X POST https://your-domain/ats/register \
  -H "Authorization: Bearer $MANIFOLD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tool_id": "payment_gateway_v2", "org_id": "acme", "domain": "finance"}'
```

---

### POST /ats/signal
Submit a trust signal (success, failure, anomaly, etc.) for a registered tool.

**Auth:** Bearer token required

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_id` | string | ✅ | Target tool |
| `signal_type` | string | ✅ | One of: `success`, `failure`, `anomaly`, `timeout` |
| `domain` | string | | Domain context |
| `stakes` | float | | Stakes of the task 0–1 |
| `submitter_hash` | string | | Anonymised submitter ID |

**Response:**

```json
{
  "recorded": true,
  "tool_id": "payment_gateway_v2"
}
```

**Example:**

```bash
curl -X POST https://your-domain/ats/signal \
  -H "Authorization: Bearer $MANIFOLD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tool_id": "payment_gateway_v2", "signal_type": "success", "stakes": 0.8}'
```

---

### GET /dashboard
Live HTML fleet dashboard showing CI history, economy ledger, entropy, threat feed, and more.

**Auth:** Public

**Response:** `text/html` — rendered in browser.

Open `https://your-domain/dashboard` in a browser, or use the Streamlit app (`streamlit run app.py`) for a richer UI.


---

### GET /report
Live visual governance dashboard for stakeholders.

**Auth:** Public

**Query params:** none

**Response:** `text/html` — auto-refreshing page with charts and tables

Shows: decision counts, action distribution, domain breakdown, tool health, consolidated policy rules.

---

### GET /digest
Structured governance summary for automation and alerting.

**Auth:** Public

**Query params:**
- `period`: `"24h"` | `"7d"` | `"30d"` (default: `"7d"`)

**Response:** JSON

**Example:**
```bash
curl https://your-host/digest?period=7d
```

Returns: `{generated_at, period, version, summary, domains, tools, policy, governance}`


---

### GET /
Landing page. Public. Returns HTML.

---

### GET /signup
Signup form. Public. Returns HTML.

---

### POST /signup
Create account. Public.

**Body:** `{ "email": "...", "org_name": "...", "domain": "general" }`

**Response:** HTML page showing the generated API key (shown once only).

---

### GET /connect
Tool connection guide. Public. Returns HTML with integration snippets for Python, LangChain, Cursor, cURL, and environment variables.

---
