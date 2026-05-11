# MANIFOLD

> **v1.7.0 | 2383 Tests Passing | Production Ready**

[![CI](https://github.com/chizoalban2003-beep/MANIFOLD/actions/workflows/manifold-ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/MANIFOLD/actions/workflows/manifold-ci.yml)
[![Tests](https://img.shields.io/badge/tests-2383%2F2383-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![Zero deps](https://img.shields.io/badge/external%20deps-0-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What is MANIFOLD?

MANIFOLD is middleware that sits between intelligent agents and the world — pricing risk before every action, learning from every outcome, and keeping humans in control of decisions that matter.

It governs **any agent**: AI models (Claude, GPT-4, Gemini, Ollama), software automation, and physical robots (Roomba, drones, robotic arms, IoT devices). It governs **any domain**: a conversation, a codebase, a legal case, a hospital ward, a factory floor, or your home. It operates at **any scale**: one developer to a fleet of physical robots.

**One change. Everything governed.**

```python
# Before — ungoverned
client = openai.OpenAI(api_key="sk-...")

# After — every call risk-priced, audited, escalation-ready
client = openai.OpenAI(
    base_url="https://your-manifold.app/v1",  # ← this is the entire change
    api_key="your-manifold-key"
)
```

---

## The CRNA Model — How Any Problem Becomes a Governed Grid

MANIFOLD encodes every problem — physical or digital — as a **CRNA grid** where each cell holds four numbers:

```
Cell(x, y, z) = {
  C: Cost        0.0-1.0  — how expensive to act here?
  R: Risk        0.0-1.0  — how dangerous is this action?
  N: Neutrality  0.0-1.0  — how unknown/uncertain is this area?
  A: Asset       0.0-1.0  — what value is available here?
}
```

**Examples of real domains as CRNA grids:**

| Domain | Cell | C | R | N | A |
|---|---|---|---|---|---|
| Home | Kitchen | 0.3 | 0.6 | 0.2 | 0.8 |
| Home | Baby's room | 0.2 | 0.95 | 0.1 | 0.7 |
| Home | Stairs | 0.85 | 0.75 | 0.05 | 0.1 |
| Code | Security module | 0.4 | 0.8 | 0.3 | 0.6 |
| Code | Dead code | 0.6 | 0.1 | 0.7 | 0.0 |
| Legal | Filed precedent | 0.3 | 0.2 | 0.2 | 0.8 |
| Legal | Jurisdiction gap | 0.5 | 0.7 | 0.9 | 0.4 |
| Finance | Liquid asset | 0.2 | 0.3 | 0.1 | 0.9 |
| Finance | Regulatory zone | 0.4 | 0.85 | 0.3 | 0.5 |

Agents navigate this grid. MANIFOLD governs every step — pricing `[C, R, N, A]`, routing to the optimal action, learning from outcomes, escalating decisions it cannot make alone.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │       YOU  (CEO / Owner)      │
                    │  Set objectives               │
                    │  Approve escalations          │
                    │  Review reports               │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │               MANIFOLD                    │
              │  ┌───────────────────────────────────┐   │
              │  │         ManifoldBrain              │   │
              │  │  CRNA 4-vector encoding            │   │
              │  │  13 actions: answer|verify|        │   │
              │  │  escalate|refuse|stop|wait|        │   │
              │  │  plan|retrieve|delegate|           │   │
              │  │  explore|exploit|clarify|use_tool  │   │
              │  └───────────────────────────────────┘   │
              │  ┌──────────────┐ ┌────────────────────┐ │
              │  │Policy Engine │ │  ATS Trust Network │ │
              │  │if/then rules │ │  Multi-org gossip  │ │
              │  │priority order│ │  Reputation ledger │ │
              │  └──────────────┘ └────────────────────┘ │
              │  ┌──────────────────────────────────────┐ │
              │  │ Real-Time Layer (v1.7.0)             │ │
              │  │ CellUpdateBus · DynamicGrid          │ │
              │  │ HealthMonitor · CRNAPlanner          │ │
              │  │ SensorBridge · NERVATURAWorld        │ │
              │  └──────────────────────────────────────┘ │
              │  ┌──────────────────────────────────────┐ │
              │  │ AgentRegistry · TaskRouter            │ │
              │  │ AgentMonitor · Vault · PolicyRules    │ │
              │  │ Federation · Brain Persistence        │ │
              │  └──────────────────────────────────────┘ │
              └───────┬────────────────┬──────────────────┘
                      │                │
          ┌───────────▼──────┐  ┌──────▼────────────────┐
          │   Digital Agents │  │   Physical Agents      │
          │  Claude · GPT-4  │  │  Roomba · Drones       │
          │  Gemini · Ollama │  │  Robotic arms · IoT    │
          │  LangChain · Any │  │  Any hardware device   │
          │  OpenAI-compat.  │  │  via manifold-physical │
          └──────────────────┘  └───────────────────────┘
```

---

## Quickstart

```bash
# Install
git clone https://github.com/chizoalban2003-beep/MANIFOLD.git
cd MANIFOLD && pip install -e .

# Run
MANIFOLD_API_KEY=your-secret python -m manifold.server --port 8080

# Sign up at http://localhost:8080/signup — get your API key
# Then point any AI tool at MANIFOLD:

export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_API_KEY="your-manifold-key"
# Done. All AI calls are now governed.
```

**LangChain:**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="your-key")
```

---

## Deploy

```bash
railway up                                           # Railway
fly launch --config deploy/fly.toml && fly deploy   # Fly.io
heroku create && git push heroku main               # Heroku
```

Environment variable: `MANIFOLD_API_KEY=your-secret`

---

## Real-Time Obstacle Handling

When something blocks an agent's path — physical or digital — MANIFOLD detects it, updates the CRNA cell values, and replans the route in real time.

### Physical obstacles (cat, person, moved furniture)

```
1. Sensor detects cat at position (3, 2, 0)
2. SensorBridge → CellUpdateBus.publish(r_delta=+0.85, ttl=30s)
3. DynamicGrid: Cell(3,2,0).R rises from 0.2 → 0.95 instantly
4. Adjacent cells R pre-raised (cat could move there)
5. CRNAPlanner: current path through (3,2,0) → invalid
6. A* replans in CRNA space → alternate route found in <50ms
7. MANIFOLD governs new path before robot moves
8. When cat leaves: R decays back to 0.2 after TTL expiry
```

### Digital obstacles (API down, rate-limited, file locked)

```
1. API returns 429 Too Many Requests
2. HealthMonitor.record_rate_limit('payments-api', retry_after=60)
3. CellUpdateBus.publish(c_delta=+0.8, r_delta=+0.5, ttl=60s)
4. DynamicGrid: tool cell C→0.9, R→0.7
5. TaskRouter: reroute sub-tasks to alternate tools or action=wait
6. After 60s: TTL expires, C/R reset to baseline, agents resume
```

### The time-extended grid

Obstacles are temporary. MANIFOLD models this as `Cell(x, y, z, t)` — cells have values that vary over time. Sometimes the optimal move is **wait**: the obstacle clears in 30 seconds and the alternate route costs more than waiting. MANIFOLD's brain already has an explicit `wait` action. The time dimension tells it when waiting beats rerouting.

---

## Governing Physical Spaces

Any physical space — home, factory, hospital, office — maps to a governed CRNA grid. Robots operating in that space navigate the grid; MANIFOLD governs every step.

```python
# floor_plan.json — describe your space as CRNA zones
from manifold_physical.space_ingestion import SpaceIngestion

ingestion = SpaceIngestion()
floorplan = ingestion.load_floorplan("my_home.json")
cells = ingestion.ingest(floorplan)
# Every room is now a governed zone
# Robots can't enter high-risk zones without MANIFOLD approval
```

```json
{
  "name": "Home",
  "rooms": [
    {
      "name": "Kitchen",
      "bounds": {"x":[0,4],"y":[0,4],"z":[0,3]},
      "crna": {"c":0.3,"r":0.6,"n":0.2,"a":0.8}
    },
    {
      "name": "Baby Room",
      "bounds": {"x":[5,8],"y":[5,8],"z":[0,3]},
      "crna": {"c":0.2,"r":0.95,"n":0.1,"a":0.7},
      "tags": ["restricted_hours_2200_0700"]
    }
  ]
}
```

Any robot registers with MANIFOLD via the same API as any AI agent:

```python
from manifold.sdk import ManifoldAgentSDK

sdk = ManifoldAgentSDK(
    agent_id="roomba-01",
    display_name="Kitchen Roomba",
    capabilities=["vacuum", "map", "floor_nav"],
    manifold_url="https://your-manifold.app",
    api_key="your-key",
    domain="physical/kitchen"
)
sdk.register()
sdk.on_command("pause",  lambda p: robot.stop())
sdk.on_command("resume", lambda p: robot.continue_task())
sdk.start_heartbeat()
sdk.start_polling()
```

---

## Policy as Code

```python
# Finance tasks at high stakes → always escalate to human
requests.post("/rules", json={
    "name": "Finance high-stakes",
    "conditions": {"domain": "finance", "stakes_gt": 0.75},
    "action": "escalate", "priority": 10
})

# Block any deletion action — digital or physical
requests.post("/rules", json={
    "name": "Block destructive actions",
    "conditions": {"prompt_regex": "\\b(delete|destroy|wipe)\\b.*\\ball\\b"},
    "action": "refuse", "priority": 100
})

# No robots in baby's room at night
requests.post("/rules", json={
    "name": "Baby room night hours",
    "conditions": {"domain": "physical/baby_room"},
    "action": "refuse", "priority": 90
})
```

Rules are evaluated before ManifoldBrain. If a rule matches, its action takes effect immediately — no risk pricing needed.

---

## The MANIFOLD World

Open `/world` in any browser. Install as a PWA on your phone.

Every agent (digital or physical) appears as an animated character in an isometric world — exactly like Clash of Clans, but the game IS your real AI and robot workforce.

- **Tap any agent** — see what it's doing, pause/resume/redirect it
- **Tap any zone** — deploy a task, set a policy
- **Watch governance beams** — fire from the MANIFOLD tower to each governed agent
- **Resource domes** — fill as work is done; harvest to trigger calibration
- **Defence sensors** — flash red when adversarial threats are detected
- **The escalation moment** — the entire world freezes, a beam fires upward toward you, your decision is required

All world state is live data from MANIFOLD via WebSocket.

---

## The NERVATURA Foundation

MANIFOLD is the implementation of **NERVATURA** — a mathematical framework where any problem becomes navigable terrain.

```
NERVATURA principle    →  MANIFOLD implementation
──────────────────────────────────────────────────
Truth as terrain       →  CRNA grid encodes reality
Agents as navigators   →  AI + robots with CRNA budgets
Governance from trade  →  ATS trust network + policy rules
Fog of war             →  N (Neutrality) decreases as agents explore
Terraforming           →  C decreases as agents improve paths
Emergent stability     →  500+ steps without central policy
```

**Experimental results from NERVATURA simulations:**
- 3.1× return on maintenance/terraforming investment
- 47.3% lower total system cost with mixed agent ecology vs single type
- 68% reduction in path uncertainty after systematic scouting
- Emergent governance stability at 500+ steps with zero hardcoded rules

---

## Key Endpoints

| Endpoint | What it does |
|---|---|
| `GET /world` | Isometric game world (PWA, mobile-ready) |
| `GET /report` | Live governance dashboard (Chart.js) |
| `GET /digest?period=7d` | JSON governance summary |
| `POST /v1/chat/completions` | Universal AI gateway — any OpenAI-compatible model |
| `POST /run` | Govern a single task with CRNA pricing |
| `POST /task` | Submit problem → decompose → route to best agents |
| `POST /agents/register` | Register any agent (AI model or physical robot) |
| `GET /agents/{id}/commands` | Long-poll — agent receives governance commands |
| `POST /rules` | Add if/then governance rule |
| `GET /plan` | A* path planning in CRNA space |
| `GET /realtime/status` | Live CellUpdateBus, DynamicGrid, HealthMonitor status |
| `GET /health/tools` | Live digital tool health summary |
| `GET /nervatura/world` | 3D CRNA voxel world engine summary |
| `GET /brain/state` | Current learning state (nodes, rules, predictions) |
| `GET /federation/status` | Multi-org ATS trust network |

Full reference → [docs/API.md](docs/API.md)

---

## Roadmap

| Version | What | Status |
|---|---|---|
| v1.6.0 | Brain persistence · Command channel · Policy rules · Federation · Game world | ✅ Done |
| v1.7.0 | CellUpdateBus · DynamicGrid · HealthMonitor · CRNAPlanner · NERVATURAWorld · Physical bridge | ✅ Done |
| v2.0.0 | MANIFOLD Physical: governed robot fleets in physical spaces | 📋 Roadmap |
| v3.0.0 | NERVATURA Platform: unified digital + physical governance OS | 📋 Vision |

---

## Numbers

| Metric | Value |
|---|---|
| Tests passing | **2383 / 2383** |
| Python modules | **86** |
| API endpoints | **48** |
| Domain packs | **7** |
| Brain actions | **13** |
| Agent types | **Unlimited** — AI and physical, same API |
| External dependencies | **0** — stdlib only for core |

---

## Contributing

```bash
git clone https://github.com/chizoalban2003-beep/MANIFOLD.git
cd MANIFOLD && pip install -e ".[dev]"
pytest tests/ -q   # 2383 tests · 0 failures · ~4 minutes
```
