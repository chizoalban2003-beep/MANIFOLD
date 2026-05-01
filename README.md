# MANIFOLD

**Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

MANIFOLD explores how a population of agents discovers, preserves, and revises
value in a changing grid. Earlier phases moved from fixed 9-box route geometry
to multi-agent, multi-objective evolution. This repository now includes the
Phase 5 baseline: ontogeny, or learning inside a single vector lifetime.

## Phase 5: finite energy armor

Each vector carries a finite battery:

```text
E_max = 30
C_total = C_base + risk_cost + energy_spent + survival_penalty
```

During traversal, a vector may spend energy on temporary armor:

- route risk above `max_risk` is lethal unless armor reduces effective risk;
- armor costs energy every timestep;
- `conserve_bias` controls how aggressively the vector spends its remaining
  budget;
- route selection compares total realized cost, so a vector may survive a spike
  by boosting armor or avoid it by taking the scout detour.

The default manifold contains three corridors:

| Route | Length | Risk behavior |
| --- | ---: | --- |
| `tank` | 5 | constant risk 6 |
| `scout` | 9 | constant risk 2 |
| `flicker` | 6 | toggles risk 3/7 every 8 generations |

Evolution still acts across generations through vector regret, conservative
mutation, and light fitness sharing. Ontogeny adds a second pressure: policies
must budget energy over the lifetime rather than only inherit good physics.

## Run the simulation

The package is intentionally dependency-light and runs with the Python standard
library.

```bash
PYTHONPATH=src python -m manifold --generations 40 --population-size 32 --seed 7
```

The command prints CSV-style generation metrics:

- average and best regret;
- survival rate;
- population diversity;
- dominant route;
- niche counts;
- average energy spent;
- average cognitive load.

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Repository layout

```text
src/manifold/ontogeny.py  Phase 5 model and experiment runner
src/manifold/__main__.py  CLI entry point
tests/test_ontogeny.py   Focused tests for energy budgeting and evolution
```

## Next experiment

After the finite battery baseline, the next MANIFOLD extension is rechargeable
sub-targets: introduce recharge cells as intermediate goals and measure whether
vectors learn hierarchical plans that trade route length for future option
value.
