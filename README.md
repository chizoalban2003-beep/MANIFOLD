# Project MANIFOLD

**MANIFOLD — Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

MANIFOLD is an evolutionary simulation where route value is discovered by a competing population rather than hardcoded in geometry. The current version includes a production-oriented stack with layered dynamics, adaptive rules, live event ingestion, marketized memory, predator auto-tuning, and explainability traces.

## Core Idea

Classical pathfinding treats center and corner utility as static heuristics.  
MANIFOLD treats those utilities as hypotheses: vectors evolve beliefs from outcomes, and must re-adapt when the world changes.

The model now spans:

1. **Geometry** (fixed route topology)
2. **Agents** (evolving physics, trust and strategy)
3. **Population × Non-stationary world** (teacher perturbations, predators)
4. **Rule/market feedback** (adaptive penalties and memory revenue)
5. **Cross-domain layers** (physical + information + social)

## Architecture Phases

`manifold/simulation.py` runs six phases:

1. `phase_1_static`
2. `phase_2_dual_niche`
3. `phase_3_teacher_flicker`
4. `phase_4_ontogeny`
5. `phase_5_recharge_hierarchical`
6. `phase_6_production_stack`:
   - 3-layer grid enabled (`physical`, `information`, `social`)
   - adaptive rule engine enabled
   - memory market enabled
   - predator auto-tuning enabled
   - confidence distribution output for human-intent targeting

## Production Extensions Implemented

### 1) Multi-layer grid dynamics
- Explicit layered risk model with cross-layer coupling:
  - info lies can propagate into physical risk
  - social signal influences information reliability
- Per-generation layer regret contributions are logged.

### 2) Memory as an asset
- Agents earn memory-market revenue for verification behavior.
- Revenue scales with verification cost and `(1 - P(lying))`, capped by reputation ceiling behavior.

### 3) Adaptive rule engine
- Rules compiled from DSL and updated each generation:
  - `penalty(t+1) = penalty(t) + alpha * (break_rate - target_rate)`
- Rule penalty histories and break rates are recorded in telemetry.

### 4) Human-intent output
- Instead of route only, the engine emits confidence distributions over cells.
- This supports decision-API style targeting.

### 5) Transfer learning artifact
- Simulation exports:
  - neutrality layer map
  - final rule penalties
  - max population reputation
- You can feed neutrality back into a new run through `run_manifold(..., transfer_neutrality_layer=...)`.

### 6) Real-time predator tuning
- Spawn rate is auto-adjusted from current max reputation:
  - above 0.85 => increase
  - below 0.70 => decrease

### 7) Explainability layer
- Decisions are logged as inequalities, e.g.:
  - `Agent 47 skipped verification at cell 4 because 0.045 < 0.08×8.20`

## Rule Compiler (DSL)

Create policy files like `manifold/sample_rules.dsl`:

```text
if late_delivery then -£8.20 @target=0.18 @alpha=1.25 @min=0.8 @max=35
if skip_verification then -£5.40 @target=0.22 @alpha=1.05 @min=0.5 @max=32
```

Compile path via CLI: `--rulebook <path>`.

## Live Data Connector

CSV format:

```text
generation,layer,cell,delta,note
4,info_noise,4,1.20,traffic rumor burst
11,physical_risk,6,0.90,road closure incident
```

Valid layer values:
- `physical_risk`
- `info_noise`
- `social_reputation`

Load path via CLI: `--connector-events <path>`.

## Predator Dashboard

`app.py` now includes:
- predator spawn rate trend
- max reputation
- death-rate context
- explainability samples
- confidence distribution

## Repository Structure

```text
.
├── app.py
├── main.py
├── manifold/
│   ├── __init__.py
│   ├── cli.py
│   ├── connectors.py
│   ├── rules.py
│   ├── sample_events.csv
│   ├── sample_rules.dsl
│   └── simulation.py
└── tests/
    └── test_simulation.py
```

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick run:

```bash
python3 main.py --quick
```

Run with production extras:

```bash
python3 main.py \
  --seed 7 \
  --rulebook manifold/sample_rules.dsl \
  --connector-events manifold/sample_events.csv \
  --output-json artifacts/manifold_seed7.json
```

Launch dashboard:

```bash
streamlit run app.py
```

## Tests

Run:

```bash
python3 -m pytest -q
```

Test coverage includes:
- ontogeny and recharge behavior
- hierarchical action usage
- rule compiler parsing
- connector CSV ingestion
- multi-layer coupling contribution
- adaptive penalty evolution
- legacy compatibility when production flags are disabled
- predator auto-tuning dynamics
- transfer artifact integrity
