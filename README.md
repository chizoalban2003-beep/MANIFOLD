# Project MANIFOLD

**Multi-Agent Non-stationary Framework for Ontogenetic Learning and Dynamic valuation**

MANIFOLD should keep its name. The project has grown from a fixed 3x3
Tic-Tac-Toe geometry into a manifold of problem spaces: any domain can be
translated into a grid of priced states, then populated by agents whose social
rules evolve from economics rather than being hand-coded.

A more current subtitle is:

> **An evolutionary engine for growing social rules from priced action.**

## Core thesis

Do not tell agents to be honest, verify sources, gossip, blacklist liars, or
forgive. Give them:

1. finite energy,
2. a vector grid,
3. priced actions,
4. brutal selection.

Then let social intelligence emerge.

## Universal mapper

Every cell is a four-component vector:

```text
[cost, risk, neutrality, asset]
```

- **Cost**: fuel, tokens, compute, money, user patience, or time.
- **Risk**: traffic, hallucination, market crash, server failure, deception.
- **Neutrality**: low-information or low-pressure space.
- **Asset**: reward, target density, credibility, order value, job reward.

The social engine runs on a 31x31 grid by default. The earlier path/teacher
engine remains available for 11x11, 21x21, and 31x31 transfer experiments.

## Social genome

Each social agent has five genes:

```text
deception     probability of lying when signalling
verification  probability of checking before trusting
gossip        probability of sharing verified risk information
memory        how long betrayal is remembered before forgiveness
energy        starting budget
```

No moral rules are programmed. Fitness is:

```text
fitness = energy_left + assets_collected - penalties
```

The top 20% reproduce; the bottom 80% die. Children inherit mutated genes.

## Emergent phases

### Generations 0-800: honeymoon
Low deception survives as random mutation. Cooperation works because lying is not
yet profitable enough to dominate.

### Generations 800-1,200: liar explosion
A deceptive mutant can get social benefit while externalizing risk. Deception can
spread until blind trust becomes expensive.

### Generations 1,200-1,456: verification phase transition
Verification becomes adaptive when:

```text
risk * penalty > verification_cost * risk_reduction
```

Using the trust-world numbers:

```text
0.45 * 0.50 = 0.225 expected loss
0.30 * 0.70 = 0.210 effective check cost
```

At that point, checking is cheaper than blind trust.

### Generation 2,000+: stable trust economy
A seeded Gen-2000 population starts near:

- deception around 32%
- verification around 54%
- gossip around 67%
- finite memory with forgiveness

Deception does not go to zero because low-stakes deception can remain cheaper
than universal checking. The stable system is not perfectly honest; it is priced.

## Blacklisting and rehabilitation

Agents remember repeated betrayal, blacklist after repeated detected lies, and
forget after a memory window. Permanent grudges are not privileged. They survive
only if they improve fitness.

Rehabilitation is priced as:

```text
forgive when expected future value > expected deception loss + verification cost
```

In practice, memory evolves into a temporary exclusion window rather than a moral
sentence. Mercy is adaptive risk pricing.

## Collusion collapse and Predatory Scouts

The monopoly acid test starts with a verification cartel controlling a large
share of traffic, charging a premium above the base physics cost of checking, and
under-reporting risk. The substrate defeats it because the base verification
cost is anchored in the environment, not in the cartel's reputation.

The collapse has two mechanisms:

1. **Routing around**: agents with high verification and low gossip become
   Scouts. When they see concentrated high-reputation sources, they stop buying
   the premium signal and perform local checks.
2. **Memory tax bankruptcy**: colluders must spend memory and reputation energy
   keeping lies consistent, which becomes metabolically expensive once traffic
   routes around them.

Predatory Scouts are now an explicit mechanism. They are not universal police;
they trigger independent verification only when source traffic is concentrated
or reputation is unusually high. This keeps verification cheap in normal regions
while making monopoly capture unstable.

## Engines in this repo

### 1. Social intelligence engine

Run the current MANIFOLD model:

```bash
python3 -m manifold --mode social --generations 120 --population 180 --seed 2500
```

Run domain presets:

```bash
python3 -m manifold --mode social --preset birmingham
python3 -m manifold --mode social --preset misinformation
python3 -m manifold --mode social --preset compute
```

### 2. Path / teacher engine

The earlier energy-budgeting world is still available:

```bash
python3 -m manifold --mode path --generations 60 --population 36 --grid-size 11
```

### Dashboard

```bash
streamlit run app.py
```

The dashboard lets you switch between the social engine and the path/teacher
engine.

## Application presets

### Birmingham delivery

- cost = fuel + time
- risk = traffic/weather
- asset = order value

Expected behavior: selective verification for high-value orders and gossip hubs
around high-information locations.

### Misinformation

- cost = time to check
- risk = believing false information
- asset = credibility

Expected behavior: high verification, muted deception, credibility as currency.

### Compute allocation

- cost = energy
- risk = server failure
- asset = job reward

Expected behavior: near-perfect health checking and load sharing when failure
penalties dwarf checking costs.

## Repository layout

```text
app.py                  Streamlit dashboard
manifold/
  __main__.py           CLI entry point
  cli.py                CLI modes for social and path engines
  simulation.py         Path/teacher MANIFOLD engine
  social.py             31x31 social-intelligence engine
tests/
  test_simulation.py    Path/teacher regression tests
  test_social.py        Social-evolution regression tests
```

## Why the name still fits

MANIFOLD no longer names a grid. It names the space of possible mappings. A city,
a conversation, a market, or a compute cluster can all be projected onto the same
cost/risk/neutrality/asset manifold. The engine does not output one perfect path.
It outputs the social contract implied by the prices.
