```
███╗   ███╗ █████╗ ███╗   ██╗██╗███████╗ ██████╗ ██╗     ██████╗
████╗ ████║██╔══██╗████╗  ██║██║██╔════╝██╔═══██╗██║     ██╔══██╗
██╔████╔██║███████║██╔██╗ ██║██║█████╗  ██║   ██║██║     ██║  ██║
██║╚██╔╝██║██╔══██║██║╚██╗██║██║██╔══╝  ██║   ██║██║     ██║  ██║
██║ ╚═╝ ██║██║  ██║██║ ╚████║██║██║     ╚██████╔╝███████╗██████╔╝
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝      ╚═════╝ ╚══════╝╚═════╝
```

### The Universal Governance OS for Digital and Physical Intelligence

> Built on **NERVATURA** — the governed intelligence framework.

[![CI](https://github.com/chizoalban2003-beep/MANIFOLD/actions/workflows/manifold-ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/MANIFOLD/actions/workflows/manifold-ci.yml)
[![Tests](https://img.shields.io/badge/tests-2383%2F2383-brightgreen)]()
[![Version](https://img.shields.io/badge/version-1.9.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![Zero deps](https://img.shields.io/badge/external%20deps-0-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What is MANIFOLD?

MANIFOLD is middleware that sits between intelligent agents and the world — pricing risk before every action using the CRNA 4-vector (Cost, Risk, Neutrality, Asset), learning from every outcome, and keeping humans in control of decisions that matter. It is not an AI model. It is the governance layer that every AI and every robot should run through before acting in the world.

It governs **any agent type**: AI language models (Claude, GPT-4, Gemini, Ollama, any OpenAI-compatible API) and physical robots (Roomba, drones, robotic arms, IoT devices, any hardware that can send an HTTP request). It governs **any domain**: finance, healthcare, legal, home, factory floor, codebase, supply chain. It operates at **any scale**: a single developer, an enterprise fleet of AI workers, or a multi-robot physical environment — all through the same governed API. Every action is risk-priced, audited, and optionally escalated before it executes.

---

## The CRNA Model

Every cell in every space — physical or digital — is encoded as a 4-vector:

```
Cell(x, y, z) = {
  C: Cost        0.0–1.0  how expensive is it to act here?
  R: Risk        0.0–1.0  how dangerous is this action?
  N: Neutrality  0.0–1.0  how uncertain or unknown is this area?
  A: Asset       0.0–1.0  what value is available here?
}
```

Agents navigate CRNA grids. MANIFOLD governs every step — pricing `[C, R, N, A]`, routing to the optimal action, learning from outcomes, and escalating decisions it cannot make alone.

**Real domain examples:**

| Domain | Cell | C | R | N | A |
|---|---|---|---|---|---|
| Home | Kitchen | 0.30 | 0.60 | 0.20 | 0.80 |
| Home | Bedroom | 0.20 | 0.40 | 0.15 | 0.70 |
| Home | Baby Room | 0.20 | 0.95 | 0.10 | 0.70 |
| Home | Stairs | 0.85 | 0.75 | 0.05 | 0.10 |
| Code | Security module | 0.40 | 0.80 | 0.30 | 0.60 |
| Legal | Jurisdiction gap | 0.50 | 0.70 | 0.90 | 0.40 |
| Finance | Regulatory zone | 0.40 | 0.85 | 0.30 | 0.50 |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/chizoalban2003-beep/MANIFOLD.git
cd MANIFOLD

# 2. Install
pip install -e .

# 3. Set API key
export MANIFOLD_API_KEY=your-secret

# 4. Run server
python -m manifold.server --port 8080

# 5. Sign up and get your key
# Open http://localhost:8080/signup in your browser
```

**Govern any OpenAI-compatible agent — one line change:**

```python
# Before — ungoverned
client = openai.OpenAI(api_key="sk-...")

# After — every call risk-priced, audited, escalation-ready
client = openai.OpenAI(
    base_url="https://your-manifold.app/v1",  # ← this is the entire change
    api_key="your-manifold-key"
)
```

**LangChain:**

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="your-manifold-key")
```

---

## Deploy

```bash
railway up                                            # Railway
fly launch --config deploy/fly.toml && fly deploy    # Fly.io
heroku create && git push heroku main                 # Heroku
```

`MANIFOLD_API_KEY` is the only required environment variable.

---

## What is Built at v1.7.0

| Core Engine | Agentic and Real-Time | Physical and World |
|---|---|---|
| ManifoldBrain — 13 actions | AgentRegistry — physical and AI agents | NERVATURAWorld — 3D voxel grid |
| PolicyRuleEngine — if/then rules | TaskRouter — cooperative decomposition | SpaceIngestion — floor plan to CRNA |
| Brain persistence across restarts | CellUpdateBus — real-time obstacle pub/sub | SensorBridge — robot events to bus |
| Shadow mode observation layer | DynamicGrid — TTL overlays | Cell occupancy and right-of-way |
| BrainBench benchmarks | DigitalHealthMonitor — API to cell updates | manifold-world — CoC isometric PWA |
| Social genome evolutionary trust | CRNAPlanner — A* with obstacle avoidance | Universal gateway POST /v1/chat/completions |
| AutoRuleDiscovery — self-writing policies | AgentMonitor — heartbeat daemon | WebSocket GET /ws |

**ManifoldBrain actions:** `answer` · `verify` · `escalate` · `refuse` · `stop` · `wait` · `plan` · `retrieve` · `delegate` · `explore` · `exploit` · `clarify` · `use_tool`

---

## Real-Time Obstacle Handling

### Physical obstacle (cat appears in robot path)

```
1. Sensor detects cat at position (3, 2, 0)
2. SensorBridge → CellUpdateBus.publish(r_delta=+0.85, ttl=30s)
3. DynamicGrid: Cell(3,2,0).R rises from 0.2 → 0.95 instantly
4. Adjacent cells: R pre-raised to model where cat could move next
5. CRNAPlanner: current path through (3,2,0) → cost exceeds risk_budget
6. A* replans in CRNA space → alternate route found in <50ms
7. MANIFOLD governs new path before robot moves
8. Cat leaves: R decays back to 0.2 after TTL expiry
```

### Digital obstacle (API rate-limited)

```
1. API returns 429 Too Many Requests
2. HealthMonitor.record_rate_limit('payments-api', retry_after=60)
3. CellUpdateBus.publish(c_delta=+0.8, r_delta=+0.5, ttl=60s)
4. DynamicGrid: tool cell C → 0.9, R → 0.7
5. TaskRouter: reroute sub-tasks to alternate tools or action=wait
6. After 60s: TTL expires, C/R reset to baseline, agents resume
```

---

## Multi-Agent Cooperation

MANIFOLD supports five cooperation patterns between any mix of physical robots and digital AI agents:

1. **Sequential handoff** — Agent A completes its zone, passes the task token to Agent B via TaskRouter.
2. **Parallel cooperation** — Multiple agents work different sub-tasks of the same goal simultaneously, each governed independently.
3. **Zone boundary handoff** — When an agent reaches the edge of its domain zone, TaskRouter assigns the next zone to a capable agent.
4. **Multi-target distribution** — TaskRouter decomposes a prompt into up to 4 sub-tasks and assigns each to the best-matched agent by capability keywords.
5. **Contact and right-of-way** — When two agents need the same cell, AgentRegistry.resolve_conflict() compares ATS health scores and sends a `yield` command to the lower-scoring agent via queue_command.

ATS (Agent Trust Score) determines priority in all conflicts. TaskRouter decomposes any problem into sub-tasks and assigns agents by capability. Any combination of physical robots and digital AI agents can cooperate toward any target configuration.

---

## Physical Space Governance

Any physical space maps to a governed CRNA grid. Define your space as JSON:

```json
{
  "rooms": [
    {
      "name": "Kitchen",
      "bounds": {"x": [0, 5], "y": [0, 4], "z": [0, 1]},
      "crna": {"c": 0.3, "r": 0.6, "n": 0.2, "a": 0.8}
    },
    {
      "name": "Baby Room",
      "bounds": {"x": [6, 10], "y": [0, 4], "z": [0, 1]},
      "crna": {"c": 0.2, "r": 0.95, "n": 0.1, "a": 0.7}
    }
  ]
}
```

Load and ingest in three lines:

```python
from manifold_physical.space_ingestion import SpaceIngestion
ingestion = SpaceIngestion()
cells = ingestion.ingest(ingestion.load_floorplan("my_home.json"))
```

Any robot registers with the same `POST /agents/register` API as any AI agent. The governance layer is identical for both.

---

## Policy as Code

Deploy governance rules at runtime with no code changes:

```bash
# Finance: escalate high-stakes decisions
curl -X POST http://localhost:8080/rules \
  -H "Authorization: Bearer $KEY" \
  -d '{"rule_id":"fin-01","org_id":"org1","name":"Finance escalation",
       "conditions":{"domain":"finance","stakes_gt":0.8},"action":"escalate","priority":100}'

# Block destructive prompts across all domains
curl -X POST http://localhost:8080/rules \
  -H "Authorization: Bearer $KEY" \
  -d '{"rule_id":"sec-01","org_id":"org1","name":"Block destructive",
       "conditions":{"prompt_contains":"delete all"},"action":"refuse","priority":200}'

# No robots in the baby room
curl -X POST http://localhost:8080/rules \
  -H "Authorization: Bearer $KEY" \
  -d '{"rule_id":"phys-01","org_id":"org1","name":"Baby room lockout",
       "conditions":{"domain":"home","risk_gt":0.9},"action":"stop","priority":150}'
```

Rules are evaluated before every brain decision. First match wins. Priority is descending.

---

## Agent SDK

Register any agent — physical or digital — in six lines:

```python
from manifold.sdk import ManifoldAgentSDK

sdk = ManifoldAgentSDK("robot-01", "Kitchen Robot", ["navigate","clean"],
                       "http://localhost:8080", "your-key", "home")
sdk.register()
sdk.on_command("yield", lambda cmd: robot.stop())
sdk.start_heartbeat()   # background thread, every 30s
sdk.start_polling()     # long-polls /agents/{id}/commands and dispatches
```

---

## The NERVATURA Foundation

NERVATURA is the theoretical framework underlying MANIFOLD. It models any intelligence — digital or physical — as agents operating in a governed CRNA voxel space. Each agent type has a natural role in the grid:

| Agent Type | Role in NERVATURA | MANIFOLD Equivalent |
|---|---|---|
| Scout | Reduces Neutrality — explores unknown cells | `explore` action |
| Miner | Extracts Asset — harvests value from cells | `exploit` action |
| Builder | Reduces Cost — improves infrastructure | `plan` + `use_tool` |
| Trader | Moves Asset — redistributes value between zones | `delegate` action |
| Guard | Tests Risk — validates safety of cells | `verify` + `refuse` |

**Experimental results from NERVATURA simulations:**
- **3.1× terraforming ROI** — mixed agent ecologies outperform single-type fleets
- **47.3% lower system cost** with Scout-Miner-Builder-Guard ecology vs pure Miner fleet
- **Emergent governance stability** at 500 steps with zero central policy — agents self-organise into stable risk-aware patterns through ATS trust signals alone

MANIFOLD is the production implementation of NERVATURA principles, extended to real AI models and real physical robots operating in real spaces.

---

## Key Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Landing page |
| GET | `/signup` | Sign-up form |
| POST | `/signup` | Create org and API key |
| GET | `/dashboard` | Governance dashboard (auth) |
| GET | `/report` | HTML analytics dashboard |
| GET | `/digest` | JSON governance summary (`?period=7d`) |
| POST | `/run` | Governed action execution (auth) |
| GET | `/learned` | Learned rules and patterns |
| POST | `/v1/chat/completions` | Universal AI gateway (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| POST | `/agents/register` | Register an agent (AI or robot) |
| GET | `/agents` | List all registered agents |
| POST | `/agents/{id}/heartbeat` | Agent heartbeat |
| POST | `/agents/{id}/pause` | Pause an agent |
| POST | `/agents/{id}/resume` | Resume an agent |
| GET | `/agents/{id}/commands` | Poll pending commands |
| POST | `/task` | Submit a task for decomposition and routing |
| GET | `/rules` | List policy rules |
| POST | `/rules` | Add a policy rule |
| DELETE | `/rules/{id}` | Delete a policy rule |
| GET | `/brain/state` | Brain persistence state and node counts |
| GET | `/federation/status` | Federation gossip network status |
| POST | `/federation/join` | Join a federation |
| POST | `/federation/gossip` | Ingest a gossip snapshot |
| GET | `/health/tools` | Tool error rates and health status |
| GET | `/plan` | A* CRNA path plan (`?sx=0&sy=0&sz=0&tx=5&ty=5&tz=0`) |
| GET | `/realtime/status` | Real-time layer status |
| GET | `/grid/occupancy` | Current cell occupancy map |
| GET | `/nervatura/world` | NERVATURAWorld JSON state |
| POST | `/nervatura/world/init` | Initialise NERVATURAWorld dimensions |
| GET | `/world` | Isometric PWA game world (HTML) |
| GET | `/world/manifest.json` | PWA manifest |
| GET | `/ws` | WebSocket — governance events, agent updates, world stats |
| GET | `/ats/score/{id}` | Agent Trust Score |
| GET | `/ats/leaderboard` | ATS leaderboard |
| POST | `/ats/register` | Register tool for ATS |
| POST | `/ats/signal` | Submit trust signal |
| POST | `/orgs` | Create organisation |
| POST | `/orgs/{id}/keys` | Add API key to org |
| POST | `/orgs/{id}/policy` | Set org-level policy |
| GET | `/admin` | Admin overview |
| GET | `/connect` | Integration connection guide |

---

## TypeScript and Node.js

```typescript
import { ManifoldClient } from "manifold-ts";

const client = new ManifoldClient({ baseUrl: "http://localhost:8080", apiKey: "your-key" });
const result = await client.run({ prompt: "Refund customer #4821", domain: "finance" });
const chat   = await client.chatCompletion([{ role: "user", content: "Summarise this report" }]);
const status = await client.worldStatus();
```

See `manifold-ts/README.md` for full TypeScript documentation.

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| **v1.7.0** | ✅ Done | CRNA engine, ManifoldBrain (13 actions), PolicyRuleEngine, brain persistence, AgentRegistry, TaskRouter, CellUpdateBus, DynamicGrid (TTL), DigitalHealthMonitor, CRNAPlanner (A*), NERVATURAWorld (3D voxel), SpaceIngestion, SensorBridge, cell occupancy and right-of-way, manifold-world (CoC PWA), universal AI gateway, WebSocket, TypeScript client, federation, ATS trust network |
| **Phase 1** | 🔄 In progress | Deploy to Railway/Fly/Heroku, onboard pilot orgs, real governance data collection, manifold-world as installable PWA on phone |
| **Phase 2** | 📋 Roadmap | MANIFOLD Physical v0.1 — Roomba bridge with real hardware, MQTT IoT connector, camera-based obstacle detection pipeline |
| **Phase 3** | 🔭 Vision | NERVATURA platform — digital + physical governance OS, brand restructure, commercial partnerships, managed cloud offering |

---

## Numbers

| Metric | Value |
|---|---|
| Tests | 2383 / 2383 ✅ |
| Python modules | 90+ |
| API endpoints | 48+ |
| Domain packs | 7 |
| Brain actions | 13 |
| Agent types | Unlimited |
| External dependencies | **0** |

---

## Contributing

```bash
git clone https://github.com/chizoalban2003-beep/MANIFOLD.git
pip install -e ".[dev]"
pytest tests/ -q
```

PRs welcome. Open an issue first for significant changes.

---

*MANIFOLD — The Universal Governance OS for Digital and Physical Intelligence.*
*Built on NERVATURA. MIT Licence. Built by Alban Chigozirim.*
