# Changelog

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
