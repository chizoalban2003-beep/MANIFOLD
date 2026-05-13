# Changelog

## [2.2.0] — Gap closures (research-grounded)

### Closed (engineering)
- **Formal Shannon entropy N**: `NERVATURACell.belief` dict + `update_belief()` / `formal_n()` / `sync_n()`.
  N is now H(P)/log(5) over a 5-state categorical belief. Research confirmed N_formal=0.8078 ≈ intuitive scalar 0.80.
- **Vault-learned transition model**: `VaultTransitionModel` in `manifold/world_model.py` replaces hardcoded 20% hazard heuristic in `WorldModelSimulator`. Falls back to 0.20 when < 50 vault events.
- **Bounded adversarial Minimax**: `AdversarialMinimax` class in `manifold/adversarial.py`. Research: refuse=optimal (0.20 damage vs 0.90 allow). Integrated into `NashEquilibriumGate.check(context=...)`.
- **Statistical convergence**: `ConvergenceMonitor` now runs Mann-Kendall (scipy.stats.kendalltau) and ADF (statsmodels) tests. Fixed Lyapunov V uses equilibrium frozen at step 200. Research: τ=−0.979, p=5.6e-235 (MK); ADF p=0.997 (still converging at 500 steps on 8×8).
- **Theory of mind L1**: `AgentRegistry.predict_agent_action()` and `predict_all_agents()` — agent intention inference from episode history. `GET /agents/predictions?zone=X`.
- **VCG task auction**: `VCGAuction` in `manifold/vcg_auction.py` — provably truthful mechanism. `use_vcg=True` in `TaskRouter`. `POST /auction`.
- **Kinodynamic planning for Roomba**: `KinodynamicPlanner` in `manifold/kinodynamic_planner.py` — physics-aware A* with heading-change cost (0.5 m/s, 2.0 rad/s). `GET /plan/roomba`.
- **Simplified BFT for federation**: `FederatedGossipBridge.receive_signed_update()` + `broadcast_signed_update()` — f+1=2 quorum voting on ATS trust scores using existing crypto.py HMAC signing.

### New endpoints (v2.2.0)
- `GET /world-model/stats` — VaultTransitionModel learning status
- `GET /agents/predictions?zone=X` — theory of mind L1 predictions
- `GET /plan/roomba` — kinodynamic path planning
- `POST /auction` — VCG task auction

### Still open (research or hardware required)
- Formal Lyapunov convergence proof (needs mathematician)
- Full Nash equilibrium (needs adversary type distribution)
- Recursive theory of mind (I-POMDP — research level)
- Full SLAM (needs physical robot + ROS2 deployment)
- Full Raft consensus (needs multi-node infrastructure)



### Fixed
- DynamicGrid Bayesian sensor fusion (replaces max-override, 4131x MSE improvement)
- TaskRouter parallel task detection via "and"/"while" connectors (100% accuracy)

### Added
- Per-agent episodic memory with domain risk estimation (42% risk reduction)
- MPCPlanner — look-ahead path planning for high-stakes physical agents
- CBSPlanner — Conflict-Based Search multi-agent pathfinding (72% conflict reduction)
- ConvergenceMonitor — real-time NERVATURA stability tracking
- `GET /plan/multi`, `GET /agents/best`, `GET /nervatura/convergence`
- `GET /agents/{id}/episodes`, `POST /agents/{id}/episodes`
- V(t) sparkline in MANIFOLD World UI (bottom-right convergence indicator)

### Validated by experiments
- EXP1 MPC: 100% win rate on risk over 50 trials
- EXP2 Bayesian: 4131x MSE improvement over max-override
- EXP3 CBS: 72.2% conflict reduction over ATS right-of-way
- EXP5 NERVATURA: V(t) 39.5% reduction, converges at step 394/500
- EXP6 Temporal: 100% task ordering accuracy
- EXP7 Episodic: 42% risk reduction, 40/40 correct assignments

## [2.0.0] — v2.0.0 Release

### Added
- `ManifoldPlugin` universal plugin standard (`manifold/plugin.py`) — `ManifoldPlugin` ABC + `PluginRegistry`
- `ManifoldAgentSDK` TypeScript class (`manifold-ts/src/agent.ts`) — register, heartbeat, command polling, start/stop lifecycle
- `scripts/demo.py` — one-command demo for the full MANIFOLD stack
- `manifold/routes/` — thin route modules splitting `server.py` into `agents`, `governance`, `federation`, `ingestion`, `physical`, `world`
- pyflakes CI gate in `.github/workflows/manifold-ci.yml` — zero-warning policy enforced on every push
- Full test suite (including `tests/test_db.py`) now runs in CI via `pip install -e ".[dev,db]"`
- `aiosqlite>=0.19` added to `[dev]` and `[db]` optional dependencies in `pyproject.toml`
- 16 unused imports removed across 9 Python files

## [1.9.0] — ManifoldLLM + Universal Policy Ingestion

### Added
- `POST /llm/chat` — natural language governance configuration via ManifoldLLM
- `GET /llm/history` — last 20 LLM exchanges
- `PolicyTranslator` — validates LLM-generated policy dicts + vocabulary mapping
- `PolicyTranslator.hipaa_preset()` / `gdpr_preset()` / `sox_preset()` / `iso27001_preset()` — industry compliance rule presets
- `POST /rules/preset` — apply a compliance preset in one call
- `GovernanceReporter` — plain English governance summaries, weekly digest, escalation explainer, policy simulation
- `GET /digest?format=text` — plain text weekly digest via GovernanceReporter
- `DocumentIngester` — ingest PDF, Markdown, CSV, TSV, plain text, and URLs into PolicyRules
- `POST /ingest/document` — ingest a URL or text body
- `ImageIngester` — floor plan images → SpaceIngestion CRNA grid; whiteboard images → PolicyRules
- `POST /ingest/image` — ingest base64-encoded image
- `AudioIngester` — voice instructions → PolicyRules (Whisper + LLM fallback)
- `POST /ingest/audio` — ingest base64-encoded audio
- `UniversalIngester` — auto-detect any input format and route to correct ingester
- `POST /ingest` — universal endpoint, accepts URL or text
- `GET /ingest/history` — last 20 ingestion events
- Model-agnostic: Claude, GPT-4V, Gemini, or local Ollama via MANIFOLD gateway

## [1.8.0] — MANIFOLD Physical v0.1

### Added
- `manifold_physical/bridges/roomba_bridge.py` — `RoombaBridge`: iRobot Roomba
  governed by MANIFOLD via iRobot REST cloud API. Command poller (every 20 s),
  sensor poller (every 5 s), bump → obstacle event, `mock_mode` for hardware-free
  testing.
- `manifold_physical/bridges/mqtt_bridge.py` — `MQTTBridge`: any MQTT 3.1.1 IoT
  device as a governed MANIFOLD agent. Minimal MQTT client using stdlib `socket`
  + `struct` only (zero external deps). `DeviceMapping` dataclass,
  `HomeAssistantProfile` classmethod for Home Assistant topics.
- `manifold_physical/camera_detector.py` — `CameraDetector`: YOLOv8 real-time
  obstacle detection to CRNA grid. Graceful fallback to motion-detection-only
  mode when `ultralytics` is not installed. `CameraRegistry` singleton.
  `RaspberryPiConfig` classmethod for Pi 5.
- `manifold_physical/physical_manager.py` — `PhysicalManager`: unified physical
  layer manager. `start_all()`, `stop_all()`, `status()`. Initialises from a
  config dict.
- `GET /physical/status` — `PhysicalManager.status()` if initialised, else empty
  shape.
- `POST /physical/init` — initialise `PhysicalManager` from request body config.
- `GET /physical/cameras` — `CameraRegistry` status list.
- `manifold_physical/config_example.json` — complete home config (Roomba + MQTT +
  camera).
- `manifold_physical/QUICKSTART.md` — step-by-step guide for first physical
  deployment (Roomba, Home Assistant, webcam).
- 16 new tests across `test_roomba_bridge.py`, `test_mqtt_bridge.py`,
  `test_camera_detector.py`, `test_physical_integration.py`.
- Full `mock_mode` support — all bridges testable without hardware.

### Changed
- Version bumped to 1.8.0 in `manifold/__init__.py` and `pyproject.toml`.
- `manifold_physical/__init__.py` exports `RoombaBridgeFull`, `MQTTBridge`,
  `DeviceMapping`, `CameraDetector`, `CameraRegistry`, `Detection`,
  `get_camera_registry`, `PhysicalManager`.
- `pyproject.toml`: added `manifold_physical.bridges` to `packages`.

## [1.7.0] — 2026-05-11

### Added — Real-Time Obstacle Handling + NERVATURA Engine
- `manifold/cell_update_bus.py` — `CellUpdateBus` pub/sub for live CRNA cell updates
- `manifold/dynamic_grid.py` — `DynamicGrid`: real-time CRNA overlay with TTL overrides
- `manifold/health_monitor.py` — `DigitalHealthMonitor`: live API/tool health → cell updates
- `manifold/planner.py` — `CRNAPlanner`: A* path planning in CRNA space with obstacle avoidance
- `manifold/nervatura_world.py` — `NERVATURAWorld`: full 3-D CRNA voxel grid engine
- `manifold_physical/space_ingestion.py` — floor plan JSON → CRNA grid population
- `manifold_physical/sensor_bridge.py` — physical sensor/robot events → CellUpdateBus
- `GET /realtime/status` — live bus, grid, health, planner status
- `GET /plan` — CRNA path planning endpoint
- `GET /health/tools` — live tool health summary
- `GET /nervatura/world` + `POST /nervatura/world/init` — world engine endpoints
- 35 new tests across 8 new test modules (all pass)
- Integration tests: obstacle blocks path, rate-limit raises C, obstacle clears

### Fixed
- `pyproject.toml`: removed deprecated `License :: OSI Approved :: MIT License` classifier
  that prevented `pip install -e .` on setuptools ≥ 76

## [1.6.0] — 2026-05-11

### Fixed
- Brain state now persists across restarts (CognitiveMap, ToolCooccurrenceGraph,
  PredictiveBrain, MemoryConsolidator) — atexit save + startup rehydration

### Added
- Bidirectional agent command channel: `queue_command`/`poll_commands` in
  `AgentRegistry`; `GET /agents/{id}/commands` (long-poll) and
  `POST /agents/{id}/command` endpoints in server
- `manifold/sdk.py`: `ManifoldAgentSDK` — stdlib-only drop-in SDK for agent
  processes (register, heartbeat, command polling)
- `manifold/policy_rules.py`: `PolicyRule` dataclass + `PolicyRuleEngine` —
  if/then policy rules with priority ordering, save/load, and pipeline integration
- `GET /rules`, `POST /rules`, `DELETE /rules/{id}` server endpoints
- `GET /brain/state` endpoint exposing brain persistence status
- Federation activation: `GET /federation/status`, `POST /federation/join`,
  `POST /federation/gossip` endpoints; background org sync thread every 300s
- `tests/test_brain_persistence.py` — 10 persistence round-trip tests
- `tests/test_command_channel.py` — 8 command channel tests
- `tests/test_policy_rules.py` — 10 policy rule engine tests

## [1.5.9] — 2026-05-08

### Added
- manifold-world/ complete CoC-quality isometric game world
  - 16×16 terrain with water shimmer+ripples, rock clusters, grass tufts, dirt path textures
  - All buildings with 5 visual levels: domain houses (multi-floor, windows, flags, torches,
    satellite structures, domain symbols), calibrator (spinning gear, exhaust pipes, energy rings),
    MANIFOLD tower (orb, buttresses, orbiting crystals), defence buildings (antenna, radar, rings, gem)
  - 6 fully animated agents: walk cycles, thought bubbles, progress bars, motion trails, level stars,
    path lines, selection rings, domain-bounded wandering, task assignment, work, return-home cycle
  - Particle system: harvest coins, completion stars, levelup sparks, breach fragments
  - Floating numbers: +tokens, +XP, -HP on every event
  - Ambient animation: waving flags, torch flames, water shimmer+ripples, resource drip, memory crystals,
    orbiting sparks around MANIFOLD orb
  - Long-press drag editor: reposition defence buildings + domain houses, live range preview,
    valid/invalid drop indicators, ghost at original position
  - Offline ticking: resources accumulate while app closed (capped at 4 hours)
  - Construction scaffolding + grow animation on building upgrade
  - WebSocket live data from MANIFOLD server (/ws) with auto-reconnect
  - REST polling fallback (/agents endpoint every 30s)
  - API key setup screen on first launch (or skip to demo mode)
  - Toast notification queue (max 2 simultaneous, slide in from bottom-right)
  - GET /world serves world from server (already present in server.py)
  - PWA manifest meta tags for mobile install

## [1.5.8] — 2026-05-08

### Added
- MANIFOLD World is now functionally built like Clash of Clans.
  Every building serves a real governance function. Economy is
  interconnected. Time is real. Consequences are real.
- World State Engine: persistent localStorage state (manifold_world_state)
  for all buildings, resources, agents, upgrades. Auto-saved every 10s.
- Tower Levels (1-5): gates agent count, zones, calibrators, heroes.
- Real resource economy: 5 token types (finance_tokens, compute_credits,
  compliance_tokens, audit_credits, manifold_energy), finite pools,
  generation rates scale with building level, depletion throttles agents.
- Calibrator (The Builder): one upgrade at a time, real countdown timers
  shown above building, scaffolding animation while upgrading, speed-up
  with manifold_energy.
- Upgrade system: domain houses and tower upgradeable with costs and timers.
  Building tap panel shows cost/time/progress and speed-up option.
- Defence Buildings: 4 functional buildings with real detection ranges —
  Inject Detector (injection, red), Anomaly Tower (degradation, amber),
  Gossip Relay (poisoning, purple), Honeypot (one-use trap, green).
- Attack system: adversarial attacks spawn at map edges, move toward tower,
  get intercepted by defences matching their type. Breach costs 50 domain
  tokens and drains that zone's agents.
- Agent Training: warmup periods tracked in WS state, level-up queue,
  hero agents (Claude, GPT-4o) with energy bars unlocked at Tower level 4.
- Base Layout Editor: toggle edit mode to drag-and-drop defence buildings,
  with real-time range preview and overlap highlighting.
- Full economy loop: tasks→domain resources, defence→manifold_energy,
  world_health→governance quality, tower_level→zone activation.
- Economy summary panel: tap centre to see income/expense/net rates.
- Web Audio sound effects for harvest, upgrade, attack, breach events.

## [1.5.7] — 2026-05-08

### Added
- Resource system: token pools per domain (finance/devops/healthcare/legal), fill rate,
  HUD resource bars with amber warning at <20%, harvest mechanic (click dome when >=80%),
  agent speed throttle when pool hits 0, floating harvest text, calibration signal to server.
- Agent levelling: health score maps to level 1-5, coloured star indicators above head
  (gold/silver/bronze/grey), zone access rules, level-up particle burst + floater text.
- Agent moods: health score drives speed, bob amplitude, colour brightness, head droop,
  stressed "!" indicator, sparkle particles at >85% health; mood bar in agent tap panel.
- Defence events: 4-phase adversarial probe sequence (shockwave, agents flee home,
  multi-beam governance fire, all-clear); automatic demo timer every 45-90s;
  triggered by WS governance_event with risk_score > 0.88 + refuse action.
- Risk weather system: ambient risk level from agent scores drives visual overlays —
  amber haze (medium), drifting storm clouds (high), lightning flashes (critical).
- The escalation moment: world freezes, dim overlay, MANIFOLD tower fires beam upward,
  decision card appears (Approve / Reject), world resumes with outcome applied.
  Demo timer every 90-150s; WebSocket escalate action also triggers it.
- Memory landscape: completed tasks spawn persistent crystals in localStorage (max 200),
  age-based alpha, tap to show label/timestamp tooltip, forms visual history over time.
- Social links: co-occurrence tracking between busy agents, dotted lines drawn between
  connected pairs (alpha scales with count), red for paused pairs, solid when collaborating,
  connections section in agent tap panel.
- Base growth: building levels 1-5 in localStorage keyed by domain task counts
  (5/15/35/75 tasks), visual upgrades per level (extra floor, roof, windows, flag, outpost),
  level-up construction animation with floater text.
- Achievement system: 8 governance milestones checked after every significant event,
  slide-in toast notification with icon, full achievement grid in MANIFOLD tower panel.
- World growth decoration: corner posts at level 4+, animated waving zone flags at level 5.
- Polish: camera lerp (smooth pan to events), ambient particles per zone (floating upward),
  particle bursts on task completion + level-up, floater texts, haptics on mobile
  (vibrate API for task complete / escalation / defence), Web Audio beep on task complete.


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
