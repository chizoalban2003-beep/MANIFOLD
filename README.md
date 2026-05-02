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

### 0. MANIFOLD Brain

MANIFOLD Brain is the broader adaptive decision layer. It sits above agents,
tools, memory, and task goals, then decides the next action:

```text
answer | clarify | retrieve | verify | use_tool | delegate | plan | explore |
exploit | wait | escalate | refuse | stop
```

Example:

```bash
python3 -m manifold \
  --mode brain \
  --prompt "Use the best tool to solve this task safely" \
  --domain coding \
  --uncertainty 0.55 \
  --complexity 0.85 \
  --stakes 0.65 \
  --tool-relevance 0.8
```

BrainBench compares MANIFOLD Brain against common agentic baselines:

```bash
python3 -m manifold --mode brainbench --generations 10 --population 32 --grid-size 5
```

Baselines include always-answer, ReAct-style, tool-first, retrieve-first, and a
static risk policy. This is the competitive testbed for positioning MANIFOLD as
an agent "executive function" rather than another generator.

BrainBench can also evaluate real agent or LLM task logs:

```bash
python3 -m manifold --mode brainbench --tasks-path data/agent_logs.csv
```

CSV schema:

```text
prompt,expected_action,domain,uncertainty,complexity,stakes,source_confidence,
tool_relevance,time_pressure,safety_sensitivity,collaboration_value,
user_patience,dynamic_goal,weight
```

Only `prompt` and `expected_action` are required. The numeric fields can be
derived from production telemetry: model confidence, retrieval confidence, tool
failure rate, latency, user tier, severity, safety classifier scores, or human
review labels.

## How MANIFOLD translates to LLMs and generative AI

MANIFOLD is not a replacement for an LLM. It does not generate prose, code, or
images. Its plausible role is the **executive decision layer** around generative
systems:

```text
LLM / tool / agent produces candidate capability
MANIFOLD Brain decides when to use, verify, defer, escalate, or refuse
```

What is plausible today:

- routing between answer, retrieve, verify, clarify, use_tool, plan, escalate,
  and refuse;
- turning task logs into cost/risk/asset benchmark rows;
- learning domain-level policy adjustments from outcomes;
- tracking tool/source reputation and over-reliance;
- benchmarking against ReAct-style, retrieve-first, tool-first, static-risk, and
  always-answer baselines.

What is still theoretical or early-stage:

- claiming open-ended general intelligence;
- replacing gradient-trained LLM reasoning;
- proving universal optimality across arbitrary problem maps;
- discovering high-quality cell vectors without telemetry or human labels;
- robustly learning from subjective outcomes unless they are mapped to measurable
  proxies such as churn, satisfaction, resolution rate, incident severity, or
  human-review labels.

The honest product claim is narrower and stronger:

> MANIFOLD Brain is an adaptive policy compiler for AI agents. It maps a task
> into measurable cost/risk/asset features and chooses the next action more
> safely and economically than naive fixed policies.

### 1. TrustRouter

TrustRouter is the first niche product built on MANIFOLD. It maps an AI-agent or
dialogue task into GridMapper OS and returns an action policy:

```text
answer | clarify | retrieve | verify | escalate | refuse
```

Example:

```bash
python3 -m manifold \
  --mode trustrouter \
  --prompt "The user asks for regulated advice from uncertain sources" \
  --domain regulated \
  --uncertainty 0.9 \
  --complexity 0.8 \
  --stakes 0.95 \
  --source-confidence 0.2 \
  --safety-sensitivity 0.9 \
  --dynamic-intent
```

The TrustRouter memory tracks prior worlds by domain, so repeated support,
legal, medical, or technical tasks can nudge future risk thresholds without
hardcoding a static policy.

TrustBench compares TrustRouter against static baselines:

```bash
python3 -m manifold --mode trustbench --generations 12 --population 32 --grid-size 5
```

The benchmark reports utility, accuracy, action cost, risk penalty, missed
verification, and unnecessary verification for:

- TrustRouter
- always answer
- clarify if uncertain
- retrieve when source confidence is low
- refuse high-safety tasks
- a hand-tuned static risk policy

Use this mode to prove whether TrustRouter is actually competitive before
deploying it into an LLM or agent workflow.

### 2. GridMapper OS

GridMapper OS is the reusable intelligent-system layer. It lets you define a
problem directly as cells, targets, and rules, then drops in the gen-2000
population:

```python
from manifold import AgentPopulation, GridWorld

world = GridWorld(size=31)
world.load_from_traffic_csv("Birmingham_Real_Grid.csv")
world.add_dynamic_targets([
    {"id": "order_1", "pos": (12, 8), "asset": 45, "moves": "random_walk"},
    {"id": "order_2", "pos": (24, 19), "asset": 22, "moves": "static"},
])
world.add_rule("late_delivery", penalty=8.2, triggers="miss_target")
world.add_rule("false_traffic_report", penalty=0.5, triggers="deception_detected")

result = AgentPopulation(seed="gen2000", n=200, predators=0.05).optimize(
    world,
    generations=300,
)
print(result.verification)
print(result.reputation_cap)
```

The CLI can run the same layer:

```bash
python3 -m manifold \
  --mode gridmapper \
  --grid-size 31 \
  --data-path data/birmingham_week.csv \
  --target order_1,12,8,45,random_walk \
  --target order_2,24,19,22,static \
  --rule late_delivery,8.2,miss_target
```

### 3. Social intelligence engine

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

Run against a real mapper CSV:

```bash
python3 -m manifold --mode social --preset birmingham --grid-size 31 --data-path data/birmingham_week.csv
```

The CSV schema is:

```text
row,col,cost,risk,asset,neutrality
```

`neutrality` is optional. Missing cells are filled as low-risk neutral space, so
sparse traffic, supply-chain, or telemetry extracts can be projected into the
grid without hand-painting every cell.

Predatory Scouts now carry an evolvable sixth gene, `predation_threshold`. This
lets the substrate discover whether the reputation cap is really near 0.85 under
real load instead of hardcoding that value.

### 4. Path / teacher engine

The earlier energy-budgeting world is still available:

```bash
python3 -m manifold --mode path --generations 60 --population 36 --grid-size 11
```

### Dashboard

```bash
streamlit run app.py
```

The dashboard lets you switch between MANIFOLD Brain, BrainBench, TrustRouter,
TrustBench, GridMapper OS, the social engine, and the path/teacher engine.

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
  gridmapper.py         Reusable problem-to-grid optimizer
  simulation.py         Path/teacher MANIFOLD engine
  social.py             31x31 social-intelligence engine
tests/
  test_gridmapper.py    GridMapper OS regression tests
  test_simulation.py    Path/teacher regression tests
  test_social.py        Social-evolution regression tests
```

## Why the name still fits

MANIFOLD no longer names a grid. It names the space of possible mappings. A city,
a conversation, a market, or a compute cluster can all be projected onto the same
cost/risk/neutrality/asset manifold. The engine does not output one perfect path.
It outputs the social contract implied by the prices.
