# Changelog

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
