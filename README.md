# MANIFOLD

**Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

---

## Core Thesis

Classical AI treats the 9-box grid as a fixed heuristic: the centre is worth **1/2** and corners **3/8** because they sit on 4 and 3 of the 8 winning routes. MANIFOLD treats those fractions as *hypotheses* that agents must discover, then *unlearn* when the world changes.

The project moves intelligence from the geometry into the agents, then into the interaction between agents and a changing geometry.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`MANIFOLD_V1_Static_Geometry.ipynb`](jupyter_notebooks/MANIFOLD_V1_Static_Geometry.ipynb) | 9-box grid, fixed cell values, A* pathfinder. Establishes the 1/2 and 3/8 fractions analytically. |
| [`MANIFOLD_V2_MAMO.ipynb`](jupyter_notebooks/MANIFOLD_V2_MAMO.ipynb) | Multi-Agent Multi-Objective system. Congestion, risk, cost vectors. Shows value is subjective to agent physics. |
| [`MANIFOLD_V3_Emergent.ipynb`](jupyter_notebooks/MANIFOLD_V3_Emergent.ipynb) | Full evolutionary engine. Agents discover 1/2 from selection alone. Bored Teacher prevents convergence. All 5 phases. |
| [`MANIFOLD_Phase5_Ontogeny.ipynb`](jupyter_notebooks/MANIFOLD_Phase5_Ontogeny.ipynb) | Finite energy batteries, real-time armour budgeting, rechargeable sub-targets, hierarchical planning. |

---

## Architecture Evolution

### V1 — Static Geometry
- 9 boxes, 8 routes, fixed weights
- Single agent, A\* pathfinder
- Performance = shortest path

### V2 — MAMO (Multi-Agent Multi-Objective)
- Congestion, risk, and cost as independent objective axes
- Multiple agents with different physics (`riskMultiplier`, `maxRisk`)
- Performance = efficiency under competition
- Pareto frontier is the true performance surface

### V3 — Emergent (Phylogenetic Learning)
- Hardcoded heuristics removed entirely
- Agents spawn with random physics, evolve via selection pressure
- Two regret signals drive the system:
  - **Vector Regret**: individual cost minus optimal cost → death and reproduction
  - **Grid Regret**: population average regret → Bored Teacher mutation
- Value shifts from spatial to temporal: `V(s)` becomes `V(s|t)`

### Phase 5 — Ontogeny (Within-Lifetime Learning)
- Each agent carries a finite energy battery `E_max = 30`
- Boosting armour costs energy per cell, per timestep
- Real-time budgeting decision at every step: armour vs detour
- Performance shifts from path length to *cognitive load*
- Rechargeable sub-targets enable hierarchical planning

---

## Key Mechanisms

### 1. Diverse Seed + Conservative Mutation
- Generation 0: 20–40 agents spanning the full physics space (`riskMultiplier` 0.1–2.5, `maxRisk` 2–9.5)
- σ = 0.04–0.05: children stay close to parents, forcing earned efficiency over lucky jumps

### 2. Sacrifice as Data Acquisition
- A dead vector paints high-cost pheromone on the grid
- The cost of death is the price of mapping *P(s'|s,a)*

### 3. Fitness Sharing
- Light penalty for crowding a niche prevents monoculture
- Maintains diversity around 1.1–1.3 instead of collapsing to 0.2

### 4. Bored Teacher
- 70% targeted spikes on the dominant niche
- 30% random environment mutations
- Triggers every 15 generations when Grid Regret plateaus
- Transforms value from spatial to temporal

---

## Experimental Results

| Phase | Finding |
|---|---|
| 1–3 (Baseline) | Agents discover centre-weighting (1/2) from selection alone in ~8 generations |
| 3 (Regret) | Grid regret drops ~18 → 0.34; best regret = 0 |
| 4 (Dual-niche) | Stable coexistence ~27 Tanks / ~6 Scouts; diversity 1.15 |
| 5 (Flicker) | Scouts extinct by gen 20; Tanks/Hybrids enter predator-prey cycles with 25–30 gen period |
| 5 (Diversity) | Oscillates 0.25–1.0, never flatlines — continuous adaptation |
| Ontogeny | Adaptive energy strategy recovers faster from risk spikes than phylogenetic adaptation alone |

---

## The Diversity Tax

Maintaining diversity at 1.15 instead of converging to monoculture at 0.19 costs:

```
Diversity tax = H_maintained − H_monoculture = 1.15 − 0.19 = 0.96
```

This tax buys a **genetic library for futures not yet seen**.

---

## Value as a Manifold

Each phase adds a dimension to the value function:

```
V(s)                    — V1: pure geometry
V(s, agent)             — V2: physics-dependent
V(s, agent, t)          — V3: time-dependent (non-stationary world)
V(s, agent, t, E)       — Phase 5: energy-dependent
```

The grid is no longer a container. It is a training signal on a manifold where each point's value depends on **who is measuring it, when, and how much energy they have left**.

---

## Requirements

```
numpy==1.26.1
pandas==2.1.1
matplotlib==3.8.0
```

(Full list in `requirements.txt`)

---

## Running the Notebooks

```bash
pip install -r requirements.txt
jupyter notebook jupyter_notebooks/
```

Open each notebook in sequence: V1 → V2 → V3 → Phase 5.
All cells should be run top-to-bottom. Each notebook is self-contained.
