# Changelog

## [1.5.6] — 2026-05-08

### Added
- manifold-world/ — full isometric game world (CoC+Sims style)
  16×16 grid with domain zones, trees, agent houses, memory crystals,
  resource nodes, defence sensors, task pillars, the MANIFOLD tower.
  Touch controls (pan, pinch-zoom) for mobile.
  WebSocket live connection to MANIFOLD server.
  Task deployment by tapping domain zones.
  Agent levelling, resource harvesting, escalation alerts.
  Mini-map overview. PWA manifest for mobile install.
- GET /world serves the game world from the MANIFOLD server
- GET /world/manifest.json serves the PWA manifest
- GET /ws WebSocket endpoint for real-time world updates
  Sends agent_update (every 5s), world_stats (every 30s), governance events.
- Game mechanics: resource system, agent levels, world health bar,
  notification system, task completion animations, governance beams.

## [1.5.5] — 2026-05-08

### Added
- manifold/agent_registry.py — AgentRegistry tracking all running agents
  Register, heartbeat, pause, resume, health scoring, stale detection.
- manifold/monitor.py — AgentMonitor background loop
  Proactively marks stale agents, detects unhealthy agents, logs events.
- manifold/task_router.py — TaskRouter for arbitrary task intake
  Decomposes complex problems into governed sub-tasks.
  Routes each to the best available registered agent.
  Returns an execution plan with governance decisions per sub-task.
- server.py: 6 new endpoints —
  GET  /agents                  list all registered agents
  POST /agents/register         agent announces itself
  POST /agents/{id}/heartbeat   keep-alive
  POST /agents/{id}/pause       MANIFOLD pauses a running agent
  POST /agents/{id}/resume      MANIFOLD resumes a paused agent
  POST /task                    submit any problem for governed routing
- 26 new tests. Total: 2305 passing.

### What this enables
MANIFOLD can now:
  1. Know which agents are running at any moment
  2. Proactively detect stale or unhealthy agents
  3. Pause or redirect a running agent via API
  4. Receive any complex problem, decompose it into sub-tasks,
     govern each sub-task, and route to the best available agent
  This is Mode 3: MANIFOLD as active manager, not just traffic light.

## [1.5.4] — 2026-05-07
### Added
- GET / — landing page for consumer onboarding
- GET /signup + POST /signup — self-service account creation with API key delivery
- GET /connect — tool connection guide with integration snippets
  (Python/OpenAI SDK, LangChain, Cursor/VS Code, cURL, environment variables)
- GET /report — live visual governance dashboard (Chart.js, auto-refresh 30s)
  Action distribution chart, domain breakdown, tool health table,
  consolidated rules panel. No Streamlit. Works in any browser.
- GET /digest?period=7d — structured JSON governance summary for
  automation, Slack alerts, email digests, Grafana, PagerDuty
- vscode-manifold/ — VS Code extension
  Commands: "Check selected code for risk", "Open governance dashboard"
  Auto-check on save (optional). Works with Cursor, Copilot, any VS Code AI.
- 8 new tests (test_report_digest.py). Total: 2299 passing.

## [1.5.3] — 2026-05-07

### Added
- manifold/orgs.py — OrgRegistry, OrgConfig, OrgRole, RBAC
  Multi-tenant org management with per-org policy configuration.
  Roles: admin / agent / readonly / viewer.
  Persists to orgs.json (configurable via MANIFOLD_ORGS_FILE).
- server.py — OrgRegistry wired into auth.
  POST /orgs — create org (admin only)
  PUT /orgs/{id}/policy — update org policy (admin only)
  POST /orgs/{id}/keys — generate new API key (admin only)
  GET /admin — policy management UI for non-technical operators
- 20 new tests. Total: 2271 passing.

### What this enables
An organisation can self-serve:
- Create isolated agent identities with their own API keys
- Configure per-org risk tolerance, veto threshold, and domain
- Update policies without touching code or redeploying
- Use the /admin UI without writing Python or curl commands
- Enforce role separation: admin configures, agent runs, viewer monitors

## [1.5.2] — 2026-05-07

### Added
- POST /v1/chat/completions — OpenAI-compatible AI gateway endpoint.
  Any OpenAI-compatible agent governed with a single base_url change.
  Supports: govern-only mode, full upstream forwarding, framework
  auto-detection via rosetta.py (OpenAI, LangChain, AutoGen, generic).
- GET /v1/models — OpenAI-compatible models list endpoint.
- MANIFOLD_UPSTREAM_URL and MANIFOLD_UPSTREAM_KEY env vars.
- docs/INTEGRATION.md — "Universal gateway" section.
- README.md — Gateway quickstart.
- 8 new tests. Total: 2251 passing.

### What this enables
Any agent using openai.OpenAI(), langchain_openai.ChatOpenAI(),
LlamaIndex, AutoGen, CrewAI, or any OpenAI-compatible SDK is
governed by MANIFOLD with one environment variable change:
  base_url="http://your-manifold-host/v1"

## [1.5.1] — 2026-05-07

### Added
- deploy/fly.toml — Fly.io deployment config
- deploy/render.yaml — Render deployment config
- deploy/railway.json — Railway deployment config
- README.md — "Deploy to production" section with 4 deployment paths
- docs/API.md — complete HTTP endpoint reference
- docs/INTEGRATION.md — three integration patterns with examples
- docs/DOMAIN_PACKS.md — domain pack reference and custom pack guide

### No source code changes. No new tests needed.

## [1.5.0] — 2026-05-07

### Added
- manifold/cognitive_map.py — CognitiveMap, 4D grid navigation,
  outcome memory, suggest_action()
- manifold/predictor.py — PredictiveBrain, predict_and_decide(),
  calibration_signal()
- manifold/consolidator.py — MemoryConsolidator, ConsolidatedRule,
  nightly pattern promotion
- manifold/workspace.py — GlobalWorkspace, keyword domain routing,
  competition_scores()
- manifold/cooccurrence.py — ToolCooccurrenceGraph, Jaccard correlation,
  propagate_flag()
- manifold/encoders/ — encode_any(), TimeSeriesEncoder, StructuredEncoder
- manifold/pipeline.py — ManifoldPipeline, full 6-module integration
- 52 new tests. Total: 2227 passing.
- 0 new mandatory dependencies added.

### Brain-to-MANIFOLD correspondence
- CognitiveMap     → hippocampal place cells / cognitive map
- PredictiveBrain  → predictive processing / cerebellum error signal
- MemoryConsolidator → slow-wave sleep consolidation
- GlobalWorkspace  → global workspace theory / thalamic routing
- ToolCooccurrenceGraph → Hebbian co-occurrence / synaptic linking
- encode_any()     → sensory cortex universal projection

## [1.4.0] — 2026-05-07

### Added
- manifold/encoder_v2.py — semantic prompt encoding via sentence-transformers
  with keyword fallback. encode_prompt() and encoder_backend() exported.
- manifold/domains/ — pluggable domain pack system with 7 packs:
  healthcare, finance, devops, legal, infrastructure, trading, supply_chain.
  Each pack has tuned escalation/refusal thresholds and allowed action sets.
- manifold/calibrator.py — nightly self-calibration loop. Reads outcome history
  from ManifoldDB and nudges domain thresholds toward target escalation rates.
  Fully deterministic, fully auditable. No ML required.
- manifold/anomaly.py — z-score sliding-window anomaly detector for tool
  behaviour. Replaces moving-average in adversarial detection pipeline.
- manifold/multiagent.py — MultiAgentBridge governing agent-to-agent traffic.
  Detects prompt injection, tracks per-pair trust scores, routes via ManifoldBrain.
- ShadowModeWrapper gains agent field and __call__ for cleaner wrapping.
- manifold/trust_network/ — Agent Trust Score (ATS) network: ToolRegistration,
  TrustSignal, AgentTrustScore, ATSRegistry with tier system and leaderboard.
- server.py: 4 new ATS endpoints (GET /ats/score/<id>, GET /ats/leaderboard,
  POST /ats/register, POST /ats/signal)
- scripts/validate_encoder.py — paraphrase robustness and domain sensitivity test
- 89 new tests.

### Infrastructure (v1.3.0 → carried forward)
- manifold/db.py — async SQLite/PostgreSQL persistence with WAL sync
- manifold/auth.py — bearer token auth, fail-fast on empty key
- Dockerfile + docker-compose with shared data volume
- manifold-ts/ TypeScript SDK with ShadowModeWrapper
- .github/workflows/manifold-ci.yml — test + docker-build + ts-build jobs
- GET /metrics Prometheus endpoint

### Fixed
- egg-info removed from git tracking
- docker-compose dashboard missing MANIFOLD_DB_URL
- ManifoldAuth silent start on empty key
- CI workflow ts-build job npm test script

## [1.3.0] — initial infrastructure release
