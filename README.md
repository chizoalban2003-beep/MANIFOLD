# Project MANIFOLD

> **v1.5.2 | 2251 Tests Passing | Production Ready**
>
> **The Trust Operating System for AI agents.**
> MANIFOLD prices risk before the agent acts — detecting adversarial
> tools, escalating high-stakes decisions to humans, and writing
> calibrated penalty legislation from observed outcomes.

[![CI](https://github.com/chizoalban2003-beep/MANIFOLD/actions/workflows/manifold-ci.yml/badge.svg)](https://github.com/chizoalban2003-beep/MANIFOLD/actions/workflows/manifold-ci.yml)
[![Tests](https://img.shields.io/badge/tests-2251%2F2251-brightgreen)]()
[![Dashboard](https://img.shields.io/badge/dashboard-live%20%2Fdashboard-purple)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![Zero deps](https://img.shields.io/badge/external%20deps-0-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## ⚡ 5-Minute Quickstart

```bash
git clone https://github.com/chizoalban2003-beep/MANIFOLD.git
cd MANIFOLD
pip install -e .                     # core engine (no Streamlit)
pip install -e ".[ui]"               # + dashboard
pip install -e ".[embeddings]"       # + semantic encoder
pip install -e ".[embeddings,ui,db]" # install everything

# Load a domain pack
from manifold.domains import load_domain
policy = load_domain("healthcare")  # or finance, devops, legal, etc.

# Run a shadow audit on your support logs
python deploy_shadow.py --input your_support_logs.csv --json > report.json

# Or use the synthetic demo (no data needed)
python deploy_shadow.py --tasks 200

# Launch the dashboard
streamlit run app.py
```

### Universal gateway (zero code changes to your agents)

```bash
# Run MANIFOLD as an AI gateway
docker run -d -p 8080:8080 \
  -e MANIFOLD_API_KEY=your-key \
  -e MANIFOLD_UPSTREAM_URL=https://api.openai.com/v1 \
  -e MANIFOLD_UPSTREAM_KEY=$OPENAI_API_KEY \
  manifold-ai
```

```python
# Point any OpenAI-compatible agent at MANIFOLD
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-manifold-key"
)
# Every call is now governed. Nothing else changes.
```

### What you get

MANIFOLD runs **silently alongside your existing agent** (shadow mode) and produces a JSON report:

```json
{
  "total_tasks": 200,
  "virtual_regret_saved": 9,
  "hitl_escalations": 12,
  "tool_failures_detected": 15,
  "gossip_inoculation_speed": 4,
  "adversarial_suspects": [
    {"tool_name": "billing_api", "warm_up_rate": 0.75, "post_rate": 0.0, "drop": 0.75}
  ],
  "penalty_proposals": [...]
}
```

Three numbers close the enterprise sale:

| Metric | Meaning |
|--------|---------|
| `virtual_regret_saved` | High-risk tasks MANIFOLD would have escalated that the naive agent auto-resolved |
| `gossip_inoculation_speed` | Tasks until all agents routed around the failing tool after its first failure |
| `hitl_escalations` | Decisions where risk × stakes exceeded threshold and demanded human review |

---

## Deploy to production

### Docker (self-hosted)

```bash
docker build -t manifold-ai .
docker run -d \
  -p 8080:8080 \
  -e MANIFOLD_API_KEY=your-key \
  -e MANIFOLD_DB_URL=sqlite:///data/manifold.db \
  -v ./data:/app/data \
  manifold-ai
```

### Fly.io

```bash
fly launch --config deploy/fly.toml
fly secrets set MANIFOLD_API_KEY=your-key
fly secrets set MANIFOLD_DB_URL=postgresql://...
fly deploy
```

### Render

Push repo to GitHub.
New Web Service → connect repo → select `render.yaml`.
Set `MANIFOLD_API_KEY` and `MANIFOLD_DB_URL` in the dashboard.
Deploy.

### Railway

```bash
railway login
railway init
railway up
railway variables set MANIFOLD_API_KEY=your-key
railway variables set MANIFOLD_DB_URL=$DATABASE_URL
```

### Verify your deployment

```bash
curl https://your-domain/metrics
# Should return Prometheus-format text starting with manifold_tasks_total

curl https://your-domain/learned
# Should return JSON with cognitive_map, cooccurrence, consolidation keys

curl -X POST https://your-domain/run \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test task", "stakes": 0.5}'
# Should return JSON with action, domain, risk_score keys
```

---

## Shadow Mode Deployment Guide

### Problem

LangChain / ReAct agents hallucinate, over-use tools, and waste money.
There is no observability layer that prices risk *before* an agent acts.

### Solution

MANIFOLD is the **executive prefrontal cortex** around your agents.
It does not replace the LLM — it decides *when* the LLM should act, verify, escalate, or refuse.

### Step 1 — Wrap your existing tools

```python
from manifold import ConnectorRegistry, ToolConnector, ToolProfile

registry = ConnectorRegistry()
registry.register(ToolConnector(
    name="zendesk_api",
    fn=your_zendesk_client.call,          # your existing callable
    profile=ToolProfile(
        name="zendesk_api",
        cost=0.05, latency=0.2,
        reliability=0.92, risk=0.08, asset=0.75,
    ),
))
```

### Step 2 — Wrap your brain in ShadowModeWrapper

```python
from manifold import ManifoldBrain, BrainConfig, ShadowModeWrapper

brain = ManifoldBrain(BrainConfig(), tools=registry.tool_profiles())
wrapper = ShadowModeWrapper(brain=brain)   # active=False: observe only
```

### Step 3 — Feed your task stream

```python
from manifold import BrainTask

for ticket in zendesk_export:
    task = BrainTask(
        prompt=ticket["body"],
        domain=ticket["category"],
        stakes=ticket["priority_score"],   # 0.0-1.0
        uncertainty=ticket["confidence"],
        complexity=ticket["complexity"],
        # ... other fields default to safe mid-range values
    )
    virtual_regret = wrapper.observe(task, actual_action=ticket["agent_action"])
```

### Step 4 — Read the report

```python
report = wrapper.shadow_report()
print(f"Disagreement rate:     {report['disagreement_rate']:.1%}")
print(f"Top disagreements:     {report['top_disagreement_actions'][:3]}")
```

### Step 5 — Activate when ready

```python
wrapper.activate()   # now MANIFOLD's decisions override the naive agent
```

---

### Feeding real logs via CLI

```bash
# Zendesk / Intercom CSV export
python deploy_shadow.py --input tickets_export.csv --json > audit_report.json

# LangSmith / OpenAI trace JSON
python deploy_shadow.py --input langsmith_runs.json --json > audit_report.json
```

**CSV format** (only `prompt` column is required):

```csv
prompt,domain,stakes,uncertainty,complexity,naive_action
"Invoice charged twice","billing",0.85,0.6,0.4,auto_resolve
"Cannot login","technical",0.5,0.7,0.3,send_template_email
```

**JSON format** (LangSmith traces, OpenAI logs — nested fields auto-detected):

```json
[
  {"input": "Invoice charged twice", "metadata": {"domain": "billing"}, "output": "auto_resolve"},
  {"input": "Cannot login", "run_type": "technical", "outputs": {"text": "template_email"}}
]
```

---

### Interpreting the JSON report

```bash
cat audit_report.json | python -c "
import json, sys
r = json.load(sys.stdin)
print(f'Virtual regret saved:  {r[\"virtual_regret_saved\"]} tasks')
print(f'HITL triggers:         {r[\"hitl_escalations\"]} decisions')
print(f'Gossip inoculation:    {r[\"gossip_inoculation_speed\"]} rounds')
if r['adversarial_suspects']:
    for s in r['adversarial_suspects']:
        print(f'HONEY-POT: {s[\"tool_name\"]}  drop={s[\"drop\"]:.0%}')
"
```

---

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

Run a customer-support Trust Audit with:

```bash
python3 -m manifold --mode trustaudit
```

Trust Audit models:

- **regret reduction**: always-answer cost vs MANIFOLD Brain routing cost;
- **gossip cost**: a support summary note as token/latency cost;
- **bad-tool memory**: repeated bad tool outcomes leaving reliability scars.

In the support niche, gossip is not abstract chatter. It is the cost of writing a
summary note that prevents the next agent from repeating the same mistake. If a
summary note costs `0.06` but prevents a repeated `0.80` late-delivery or
bad-tool penalty, gossip has positive expected utility.

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

### How it compares to Copilot, agentic AI, and LLM products

MANIFOLD Brain is closest to an **agent router / orchestration policy layer**,
not to a code assistant like Copilot and not to a foundation model.

| System type | What it does | MANIFOLD's relationship |
| --- | --- | --- |
| Copilot / code LLM | Generates code and text from context | MANIFOLD could decide when to use a code model, verify output, run tests, or ask for clarification |
| ReAct-style agent | Loops between reasoning and tool calls | MANIFOLD competes with the policy that decides when to reason, act, retrieve, or stop |
| RAG system | Retrieves evidence for generation | MANIFOLD decides when retrieval is worth its cost and when source confidence is too low |
| Safety router | Blocks or escalates risky tasks | MANIFOLD generalizes this into priced risk, verification, escalation, and refusal |
| Workflow orchestrator | Routes tasks through tools | MANIFOLD adds adaptive trust, reputation, and outcome learning |

So the competitive niche is not "better than an LLM at language." It is:

> Better executive judgment around LLMs, tools, agents, and human escalation.

Run bounded research probes with:

```bash
python3 -m manifold --mode research
```

These probes currently test three modest claims:

1. Bad problem maps reduce utility, so telemetry/labels matter.
2. Negative outcomes increase future domain risk pressure.
3. MANIFOLD Brain can beat naive ReAct-like, tool-first, retrieve-first, and
   always-answer baselines on bundled benchmarks.

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
# Phase 25: Live HTTP fleet dashboard (zero external dependencies)
python -m manifold.server --port 8080
# then open http://localhost:8080/dashboard

# Phase 7: Streamlit local dashboard
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
deploy_shadow.py        Trust Audit V2 CLI — feeds real or synthetic logs through all 12 phases
app.py                  Streamlit dashboard (Shadow Mode + all earlier engines)
pyproject.toml          Package config — install with: pip install -e ".[ui]"
scripts/
  deploy_oracle.sh      POSIX deploy script: Python 3.12 check, PYTHONPATH, starts server.py
manifold/
  __main__.py           CLI entry point
  cli.py                CLI modes for social and path engines
  brain.py              Phase 1-3: ManifoldBrain, HierarchicalBrain, BrainTask
  encoder.py            Phase 4-5: PromptEncoder, DualPathEncoder, SemanticBridge
  transfer.py           Phase 6: ReputationRegistry, WarmStartConfig
  server.py             Phase 7+25: Zero-dep HTTP daemon; GET /dashboard (live fleet UI)
  connector.py          Phase 8: ToolConnector, ConnectorRegistry, ShadowModeWrapper
  hitl.py               Phase 9: HITLGate, TeacherSpike, HITLConfig
  federation.py         Phase 10: FederatedGossipBridge, GlobalReputationLedger
  adversarial.py        Phase 11: AdversarialPricingDetector, NashEquilibriumGate
  autodiscovery.py      Phase 12: AutoRuleDiscovery, PenaltyOptimizer, PolicySynthesizer
  interceptor.py        Phase 13: ActiveInterceptor, @shield
  hub.py                Phase 15: CommunityBaseline, ReputationHub
  recruiter.py          Phase 17: SovereignRecruiter, MarketplaceListing
  policy.py             Phase 18: ManifoldPolicy, PolicyDomain, DOMAIN_TEMPLATES
  gitops.py             Phase 19: ManifoldCICheck, CIRiskReport, generate_github_action
  b2b.py                Phase 20: B2BRouter, AgentEconomyLedger, EconomyEntry
  crypto.py             Phase 21: PolicySigningKey, OrgPolicySigner, GossipSigner
  fleet.py              Phase 22: CIBuildHistory, B2BEconomySnapshot, FleetPanelRenderer
  polyglot.py           Phase 23: ManifoldOpenAPISpec, spec_to_json, spec_to_yaml
  vault.py              Phase 24: ManifoldVault — thread-safe WAL + state recovery
  gridmapper.py         Reusable problem-to-grid optimizer
  simulation.py         Path/teacher MANIFOLD engine
  social.py             31x31 social-intelligence engine
tests/                  775 tests covering all 25 phases
```

### Live HTTP Oracle Server (Phase 25)

```bash
# Start the server (requires Python 3.12+)
./scripts/deploy_oracle.sh --port 8080

# Or directly:
python -m manifold.server --port 8080
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard` | Live fleet dashboard (HTML, Tailwind CSS, no JS libs) |
| `GET` | `/policy` | Active `ManifoldPolicy` as JSON |
| `GET` | `/reputation/<id>` | Agent reliability score from `ReputationHub` |
| `POST` | `/shield` | Run a `BrainTask` through the `@shield` interceptor |
| `POST` | `/b2b/handshake` | B2B policy handshake via `B2BRouter` |
| `POST` | `/recruit` | Sovereign Recruiter — discover and register tools |

The server replays the Phase 24 **Write-Ahead Log** on startup to restore gossip and economy state across restarts.

## Why the name still fits

MANIFOLD no longer names a grid. It names the space of possible mappings. A city,
a conversation, a market, or a compute cluster can all be projected onto the same
cost/risk/neutrality/asset manifold. The engine does not output one perfect path.
It outputs the social contract implied by the prices.
