# Project MANIFOLD

**Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

MANIFOLD is an evolutionary simulation testing how intelligence emerges when
survival depends on budgeting finite energy against a changing environment. It
separates two learning timescales:

- **Phylogeny**: slow evolution of body parameters such as `risk_multiplier` and
  `max_risk`.
- **Ontogeny**: fast evolution of behavior parameters such as energy aversion,
  charger use, path choice, signalling, and verification.

The core cost equation is:

```text
C_total = C_terrain + C_teacher + C_waste
```

`C_terrain` is static risk, `C_teacher` is dynamic spike pressure, and `C_waste`
is energy spent inefficiently.

## Current implementation

- 11x11 default world, with scalable 21x21 and 31x31 transfer worlds
- Start `(0, 5)`, target `(10, 5)` in the default grid
- Genome traits:
  - `risk_multiplier` (`rm`) from 0.1 to 1.0
  - `max_risk` (`max_r`) from 4.0 to 6.5
  - `energy_aversion` from 0.5 to 2.5
  - charger, honesty, and verification policies
- Energy model: `E_MAX = 8`
- Boost rule: if `risk > max_r`, energy cost is `(risk - max_r) * 2`
- Two chargers at `(3, 5)` and `(7, 5)`, restoring `+4` energy by default
- Teacher modes: periodic, reactive, random, adversarial, and multi-teacher
- Optional 2-bit communication channel with deception and verification pressure
- Streamlit dashboard plus CLI runner
- Focused pytest coverage for energy, chargers, dual-rate mutation,
  communication metrics, and transfer scaling

## Quick start

```bash
pip install -r requirements.txt
python3 -m manifold --generations 60 --population 36 --seed 7
python3 -m manifold --teacher-mode multi --communication --generations 120
streamlit run app.py
```

## Experiment map

### Phase 1-2: Baseline and low battery
High battery settings produce easy survival. Reducing `E_MAX` to 8 creates
pressure, exposing the difference between conservative risk avoidance and true
budgeting.

### Phase 3: Bottleneck corridor
The center corridor has calm risk around 5 and teacher spikes that push effective
risk toward 9. Agents must either evolve higher `max_risk`, spend energy, route
through chargers, or die.

### Phase 4: Fixed physics test
Locking body traits while keeping teacher pressure demonstrates the project
thesis: ontogenetic policy cannot rescue a body that cannot physically cross the
world.

### Phase 5: Dual-track evolution
Body mutation (`max_r_sigma = 0.02`) is slow while policy mutation
(`aversion_sigma = 0.12`) is fast. This lets body tolerance move gradually while
energy aversion, charger use, and path policy adapt quickly.

### Phase 5c: Rechargeable chargers
Chargers at `(3, 5)` and `(7, 5)` allow conditional routing. Calm periods should
favor direct travel; spike periods make charger harvesting valuable enough to
support survival.

### Phase 6-7: Adversarial and multi-teacher pressure
Teacher modes can spike when charger dependence is visible, react to high
survival, or inject random spikes. Multi-teacher mode tracks competing teacher
strengths so unpredictability can dominate learned schedules.

### Phase 8: Transfer learning
Use `transfer_population(source, target_grid_size=21)` to seed a larger world
with an evolved population. Hybrid timing strategies should transfer better than
geometry-specific routes.

### Phase 9-12: Communication, deception, verification
When `communication_enabled=True`, agents emit 2-bit signals at an energy cost.
Signal `10` represents an upcoming spike window. Low honesty creates false spike
messages; verification bias models receiver skepticism.

## Repository layout

```text
app.py                  Streamlit dashboard
manifold/
  __main__.py           CLI entry point
  cli.py                CLI argument parsing and summary output
  simulation.py         MANIFOLD simulation engine
tests/
  test_simulation.py    Focused regression tests
```

The important metric is not one shortest path. It is whether the population can
maintain body diversity, behavioral plasticity, and social verification under
changing teacher pressure.
