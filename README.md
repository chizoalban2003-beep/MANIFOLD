# Project MANIFOLD

**MANIFOLD** stands for **Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**.

This repository now contains a runnable simulation and Streamlit interface for the project described in the design brief:

- a 3x3 grid with 8 traversable routes,
- heterogeneous vectors seeded across a broad physics range,
- regret-driven selection and conservative mutation,
- fitness sharing to preserve multiple niches,
- a bored teacher that perturbs the dominant corridor when adaptation plateaus,
- and a Phase 5 ontogeny layer where each vector must budget a finite energy battery during its own lifetime.

## Core idea

Classical heuristic search bakes value into the geometry. In MANIFOLD, geometry is only the starting condition. Agents must discover high-value regions through population dynamics, then keep adapting as the world changes.

The center cell in the 3x3 grid still participates in 4 of the 8 routes while each corner participates in 3 of the 8 routes. That gives the familiar static prior:

- center weight = 4 / 8 = 0.5
- corner weight = 3 / 8 = 0.375

MANIFOLD treats those values as an emergent hypothesis rather than a hardcoded rule.

## Implemented phases

### V1 - Static geometry

- Fixed route structure
- No teacher perturbations
- No lifetime energy budget
- Useful for confirming that the population can recover the center-heavy prior from selection alone

### V3 - Emergent adaptation

- Multi-agent competition over the same 8 routes
- Distinct corridor risk profiles
- Fitness sharing across scout, hybrid, and tank niches
- Death leaves pheromone-like cost traces
- A bored teacher injects targeted or random spikes when regret plateaus
- Flicker corridor alternates between low and high risk every 8 generations

### Phase 5 - Ontogeny energy budget

- Each vector carries `E_max = 30`
- Risk can be handled by spending energy on temporary armor
- Energy spent now cannot be spent later
- The decision problem shifts from shortest path to dynamic within-lifetime budgeting

This repo implements that Phase 5 baseline directly in the simulator.

## Repository layout

```text
app.py                    Streamlit application
manifold/
  __init__.py             Public package exports
  simulation.py           MANIFOLD engine and reporting helpers
tests/
  test_simulation.py      Focused regression tests
Procfile                  Streamlit entrypoint
setup.sh                  Streamlit runtime config
```

## Simulation model

The current engine uses:

- **8 routes** over the 3x3 grid (rows, columns, diagonals)
- **Generation-0 diverse seeding** across:
  - `risk_multiplier`
  - `max_risk`
  - `energy_policy`
  - `armor_efficiency`
  - `learning_rate`
  - `explore_rate`
- **Conservative mutation** controlled by `mutation_sigma`
- **Vector regret** as the gap between realized cost and the best available route in the same generation
- **Grid regret** approximated by mean population regret over time
- **Fitness sharing** via a light crowding penalty applied inside each niche
- **Teacher shocks** triggered on plateau windows
- **Death pheromone** that increases future route cost through cells where agents failed

## Running locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit interface:

```bash
streamlit run app.py
```

## Running tests

```bash
pytest
```

## Streamlit dashboard

The app exposes controls for:

- scenario selection (`geometry`, `emergent`, `ontogeny`)
- random seed
- generation count
- optional population-size override
- mutation sigma

It then visualizes:

- regret, diversity, and energy usage over time,
- mortality, flicker intensity, and active spikes,
- the static geometry prior versus final population traffic,
- final population genomes,
- route definitions,
- and teacher intervention events.

## Interpretation guide

- **Geometry scenario**: should recover the classic center-heavy occupancy prior without any hand-coded route heuristic in the agents.
- **Emergent scenario**: should show population-level adaptation under teacher shocks and changing corridor risk.
- **Ontogeny scenario**: should show non-zero energy expenditure because vectors now solve a within-lifetime budgeting problem instead of pure route-length minimization.
  In some seeds the final generation may have already adapted away from spending energy, so the more reliable indicators are peak or cumulative energy usage over the run.

## Deployment note

The existing `Procfile` launches the Streamlit app directly:

```text
web: sh setup.sh && streamlit run app.py
```

That means the repository is ready for the same style of deployment expected by the original scaffold, but it now points to a real application instead of a missing file.
