# Project MANIFOLD

**MANIFOLD — Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

MANIFOLD is an evolutionary simulation where route value is discovered by a competing population rather than hardcoded in geometry. The system starts from a classic 3x3, 8-route manifold and progressively introduces competition, environment mutation, and within-lifetime energy budgeting.

## Core Idea

Classical pathfinding treats center and corner utility as static heuristics.  
MANIFOLD treats those utilities as hypotheses: vectors evolve beliefs from outcomes, and must re-adapt when the world changes.

The project moves intelligence across three levels:

1. **Geometry** (fixed route topology)
2. **Agents** (evolving physics and route preferences)
3. **Population × Non-stationary world** (teacher-induced adaptation pressure)

## Implemented Architecture

`manifold/simulation.py` includes a full end-to-end model with four phases:

1. `phase_1_static`
   - fixed map
   - no teacher, no flicker
2. `phase_2_dual_niche`
   - introduces scout/tank corridor asymmetry
3. `phase_3_teacher_flicker`
   - enables bored teacher spikes and flickering corridor risk
4. `phase_4_ontogeny`
   - keeps non-stationarity and adds finite per-agent energy budgeting

### Mechanisms Included

- Diverse seed population across `risk_multiplier` and `max_risk`
- Conservative mutation (`sigma=0.05`) for local adaptation
- Vector regret and grid regret proxy (population average regret)
- Death pheromone feedback (deaths increase route danger signal)
- Fitness sharing to reduce niche monoculture
- Bored teacher:
  - plateau-triggered every fixed interval
  - 70% targeted route spikes, 30% random cell spikes
- Flicker corridor with periodic risk toggle
- Ontogeny:
  - finite battery (`energy_max=30`)
  - armor spending decision per cell/per timestep
  - energy spend contributes directly to total route cost

## Repository Structure

```text
.
├── app.py                   # Streamlit viewer
├── main.py                  # CLI entrypoint wrapper
├── manifold/
│   ├── __init__.py
│   ├── cli.py               # argparse CLI
│   └── simulation.py        # core engine
└── tests/
    └── test_simulation.py   # behavior tests
```

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a fast smoke simulation:

```bash
python main.py --quick
```

Run full simulation and save telemetry:

```bash
python main.py --seed 7 --output-json artifacts/manifold_seed7.json
```

Launch Streamlit app:

```bash
streamlit run app.py
```

## Tests

Run tests with:

```bash
pytest -q
```

Current tests cover:

- non-zero energy spend during ontogeny phase
- teacher events in teacher-enabled phases
- population size bounds across all generations

## Notes on Extension

The next research extension after this baseline is rechargeable sub-targets (hierarchical planning under dynamic budget replenishment), which can be integrated by:

- introducing intermediate waypoints with local reward/charge mechanics
- adding action choices beyond route selection (e.g., pause/recharge/advance)
- tracking policy quality over delayed credit horizons
