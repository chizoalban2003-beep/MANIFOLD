# Project MANIFOLD

**Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

MANIFOLD turns a fixed 3x3 route puzzle into a non-stationary learning substrate.
The grid starts as geometry, becomes an evolutionary pressure, and now asks each
agent to learn within its own lifetime by budgeting a finite energy battery.

## What is implemented

- **Phase 5 ontogeny simulation** with finite energy (`E_max = 30`)
- Evolving vector population with risk multiplier, armor tolerance, recharge bias,
  and conservation bias
- Flickering corridor risk (`3 <-> 7`) and Bored Teacher targeted mutations
- Death pheromones as negative data acquisition signals
- Optional rechargeable sub-targets for hierarchical planning experiments
- Streamlit dashboard plus a command-line runner
- Focused tests for energy budgeting, recharge behavior, and experiment output

## Quick start

```bash
pip install -r requirements.txt
python -m manifold --generations 60 --population 36 --seed 7
streamlit run app.py
```

## Core idea

Classical heuristics value a cell by fixed route geometry. MANIFOLD asks agents to
discover those values, lose them when the world changes, and preserve enough
diversity to recover capabilities that are temporarily useless.

Phase 5 moves part of that adaptation from **phylogeny** to **ontogeny**:

```text
C_total = C_base + integral(E(delta armor_t) dt)
```

Each vector has a battery. When a route spikes, it must decide whether to spend
energy boosting armor or conserve energy and take a detour. Performance is no
longer just path length; it is also cognitive load and budget discipline.

## Repository layout

```text
app.py                  Streamlit dashboard
manifold/
  __main__.py           CLI entry point
  simulation.py         MANIFOLD simulation engine
tests/
  test_simulation.py    Focused regression tests
```

## Experiments to try

1. Run the default simulation and watch hybrids appear as the flickering corridor
   punishes pure efficiency.
2. Enable rechargeable sub-targets and compare route shares: agents with higher
   recharge bias should accept a longer route when it restores budget.
3. Increase the teacher interval pressure and observe diversity oscillations.

The important metric is not a single shortest path. It is whether the population
keeps a genetic and behavioral library broad enough to survive future geometries.
