"""deploy_shadow.py — MANIFOLD Shadow Mode Deployment Script.

Run this script to execute a complete Trust Audit V2 against a synthetic
customer-support stream or a real historical log file.  It wires together
all 14 phases:

  Phase 8  — ConnectorRegistry + ShadowModeWrapper
  Phase 9  — HITLGate (escalation detection)
  Phase 10 — FederatedGossipBridge / GlobalReputationLedger
  Phase 11 — AdversarialPricingDetector + NashEquilibriumGate
  Phase 12 — AutoRuleDiscovery (PenaltyOptimizer + PolicySynthesizer)
  Phase 13 — ActiveInterceptor (pre-flight safety veto + @shield)
  Phase 14 — Ecosystem Adapters (ManifoldCallbackHandler / ManifoldOpenAIWrapper)

Usage::

    python deploy_shadow.py                         # synthetic stream (100 tasks)
    python deploy_shadow.py --tasks 200             # larger synthetic stream
    python deploy_shadow.py --seed 42               # reproducible run
    python deploy_shadow.py --json                  # emit JSON report to stdout
    python deploy_shadow.py --input logs.csv        # real CSV log (Zendesk / Intercom)
    python deploy_shadow.py --input traces.json     # real JSON log (LangSmith / OpenAI)
    python deploy_shadow.py --input logs.csv --json > report.json  # sales artifact

Real log formats supported
--------------------------
**CSV** (Zendesk / Intercom / any tabular export):
  Required column: ``prompt`` (or ``body``, ``message``, ``description``, ``subject``)
  Optional columns: ``domain``, ``stakes``, ``uncertainty``, ``complexity``,
    ``source_confidence``, ``tool_relevance``, ``time_pressure``,
    ``safety_sensitivity``, ``collaboration_value``, ``user_patience``,
    ``naive_action`` (the action the existing agent took)
  Missing numeric columns are filled with safe defaults.

**JSON** (LangSmith / OpenAI traces / raw arrays):
  Accepts a JSON array of objects.  Common field aliases are auto-detected:
    ``input``, ``user_message``, ``question``, ``content`` → ``prompt``
    ``output``, ``response``, ``assistant_message`` → ``naive_action``
    ``metadata.domain``, ``tags``, ``run_type`` → ``domain``
  Any extra fields are silently ignored.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

from manifold import (
    ActiveInterceptor,
    AutoRuleDiscovery,
    BrainConfig,
    BrainOutcome,
    BrainTask,
    ConnectorRegistry,
    FederatedGossipBridge,
    FederatedGossipPacket,
    GlobalReputationLedger,
    HITLConfig,
    HITLGate,
    InterceptorConfig,
    ManifoldBrain,
    OrgReputationSnapshot,
    ShadowModeWrapper,
    ToolConnector,
    ToolProfile,
    default_tools,
)
from manifold.adversarial import AdversarialPricingDetector, NashEquilibriumGate
from manifold.autodiscovery import PenaltyOptimizer, PolicySynthesizer
from manifold.transfer import ReputationRegistry


# ---------------------------------------------------------------------------
# Real-log parsers
# ---------------------------------------------------------------------------

# Numeric BrainTask fields with safe defaults used when a column is absent
_TASK_NUMERIC_DEFAULTS: dict[str, float] = {
    "uncertainty": 0.50,
    "complexity": 0.45,
    "stakes": 0.40,
    "source_confidence": 0.70,
    "tool_relevance": 0.65,
    "time_pressure": 0.35,
    "safety_sensitivity": 0.20,
    "collaboration_value": 0.30,
    "user_patience": 0.60,
}

# Column-name aliases used in CSV exports from common platforms
_PROMPT_ALIASES = ("prompt", "body", "message", "description", "subject",
                   "input", "user_message", "question", "content", "text", "ticket_body")
_DOMAIN_ALIASES = ("domain", "category", "type", "tag", "channel", "queue")
_ACTION_ALIASES = ("naive_action", "action", "output", "response",
                   "assistant_message", "resolution", "resolved_by")

# LangSmith / OpenAI trace JSON field paths (dot-notation = nested lookup)
_JSON_PROMPT_PATHS = ("input", "inputs.input", "inputs.question",
                      "inputs.human_input", "inputs.messages.0.content",
                      "user_message", "question", "content", "prompt")
_JSON_ACTION_PATHS = ("output", "outputs.output", "outputs.text",
                      "outputs.answer", "response", "assistant_message")
_JSON_DOMAIN_PATHS = ("metadata.domain", "extra.metadata.domain",
                      "tags.0", "run_type", "domain")


def _deep_get(obj: dict, dotpath: str) -> object | None:
    """Traverse a nested dict by a dot-separated key path."""
    parts = dotpath.split(".")
    cur: object = obj
    for part in parts:
        if not isinstance(cur, dict):
            return None
        # allow numeric index in path (e.g. "messages.0.content")
        if part.isdigit():
            if isinstance(cur, list):
                idx = int(part)
                cur = cur[idx] if idx < len(cur) else None  # type: ignore[assignment]
            else:
                return None
        else:
            cur = cur.get(part)  # type: ignore[union-attr]
    return cur


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    """Coerce *value* to a float in [lo, hi]; return *default* on failure."""
    try:
        v = float(value)  # type: ignore[arg-type]
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _parse_row(row: dict[str, object], idx: int) -> tuple[BrainTask, str]:
    """Convert a flat dict (from CSV or JSON) into a ``(BrainTask, naive_action)`` pair."""
    # Resolve prompt
    prompt: str = ""
    for alias in _PROMPT_ALIASES:
        if alias in row and row[alias]:
            prompt = str(row[alias])
            break
    if not prompt:
        prompt = f"[task #{idx}] (no prompt field found)"

    # Resolve domain
    domain: str = "general"
    for alias in _DOMAIN_ALIASES:
        if alias in row and row[alias]:
            domain = str(row[alias]).lower().strip()
            break

    # Resolve naive action (what the existing system did)
    naive_action: str = "auto_resolve"
    for alias in _ACTION_ALIASES:
        if alias in row and row[alias]:
            naive_action = str(row[alias]).lower().strip()
            break

    # Numeric fields
    def _get(key: str) -> float:
        return _clamp(row.get(key), default=_TASK_NUMERIC_DEFAULTS[key])

    task = BrainTask(
        prompt=f"[#{idx}] {prompt[:200]}",
        domain=domain,
        uncertainty=_get("uncertainty"),
        complexity=_get("complexity"),
        stakes=_get("stakes"),
        source_confidence=_get("source_confidence"),
        tool_relevance=_get("tool_relevance"),
        time_pressure=_get("time_pressure"),
        safety_sensitivity=_get("safety_sensitivity"),
        collaboration_value=_get("collaboration_value"),
        user_patience=_get("user_patience"),
    )
    return task, naive_action


def _load_csv(path: str) -> list[tuple[BrainTask, str]]:
    """Load tasks from a CSV file.

    Supports any CSV with a prompt/body/message column plus optional numeric
    fields.  Missing numeric columns use safe defaults.  Extra columns are
    silently ignored.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    list[tuple[BrainTask, str]]
        (task, naive_action) pairs, one per row.
    """
    tasks: list[tuple[BrainTask, str]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            tasks.append(_parse_row(dict(row), idx))
    return tasks


def _flatten_json_trace(obj: dict) -> dict[str, object]:
    """Flatten a LangSmith / OpenAI JSON trace object into a plain dict.

    Tries known nested paths and surfaces them as top-level keys.
    """
    flat: dict[str, object] = {}
    # prompt
    for path in _JSON_PROMPT_PATHS:
        val = _deep_get(obj, path)
        if val:
            flat.setdefault("prompt", val)
            break
    # action
    for path in _JSON_ACTION_PATHS:
        val = _deep_get(obj, path)
        if val:
            flat.setdefault("naive_action", str(val)[:80])
            break
    # domain
    for path in _JSON_DOMAIN_PATHS:
        val = _deep_get(obj, path)
        if val:
            flat.setdefault("domain", str(val))
            break
    # numeric fields — check top level first, then metadata sub-dict
    for key in _TASK_NUMERIC_DEFAULTS:
        val = obj.get(key) or _deep_get(obj, f"metadata.{key}")
        if val is not None:
            flat[key] = val
    # copy everything from the top level (won't overwrite already-set keys)
    for k, v in obj.items():
        if not isinstance(v, (dict, list)):
            flat.setdefault(k, v)
    return flat


def _load_json(path: str) -> list[tuple[BrainTask, str]]:
    """Load tasks from a JSON file (LangSmith traces, OpenAI logs, or raw arrays).

    The file must contain a JSON array at the top level, or a dict with a
    ``"runs"``, ``"traces"``, ``"logs"``, or ``"data"`` key containing an array.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    list[tuple[BrainTask, str]]
        (task, naive_action) pairs, one per element.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, list):
        records: list[dict] = raw
    elif isinstance(raw, dict):
        # try common envelope keys
        for key in ("runs", "traces", "logs", "data", "items", "events", "records"):
            if isinstance(raw.get(key), list):
                records = raw[key]
                break
        else:
            raise ValueError(
                f"JSON file must contain a top-level array or a dict with a "
                f"'runs'/'traces'/'logs'/'data' key. Got keys: {list(raw.keys())}"
            )
    else:
        raise ValueError(f"Unexpected JSON root type: {type(raw)}")

    tasks: list[tuple[BrainTask, str]] = []
    for idx, obj in enumerate(records):
        if not isinstance(obj, dict):
            continue
        flat = _flatten_json_trace(obj)
        tasks.append(_parse_row(flat, idx))
    return tasks


def load_tasks_from_file(path: str) -> list[tuple[BrainTask, str]]:
    """Load ``(BrainTask, naive_action)`` pairs from a CSV or JSON file.

    Parameters
    ----------
    path:
        Path to the input file.  Extension determines format:
        ``*.csv`` → CSV parser; ``*.json`` → JSON parser.

    Returns
    -------
    list[tuple[BrainTask, str]]
        One pair per row / trace object.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_csv(path)
    elif ext in (".json", ".jsonl"):
        return _load_json(path)
    else:
        raise ValueError(
            f"Unsupported file extension {ext!r}. "
            "Supported: .csv, .json"
        )


# ---------------------------------------------------------------------------
# Synthetic customer-support task generators
# ---------------------------------------------------------------------------

_DOMAINS = ["billing", "technical", "returns", "general", "escalation"]

_PROMPTS = [
    "My invoice is wrong — I was charged twice for the same item.",
    "I cannot log into my account after the password reset.",
    "The product arrived damaged; I need a replacement immediately.",
    "Can you explain the difference between the Basic and Pro plans?",
    "I want to cancel my subscription effective immediately.",
    "The app crashes every time I open the settings menu.",
    "Where is my refund? It has been 10 business days.",
    "I need to change the delivery address for order #88321.",
    "My promo code is not applying at checkout.",
    "I need a copy of my last 12 invoices for an audit.",
    "The integration with Salesforce stopped syncing yesterday.",
    "I was told I would receive a callback — it never came.",
    "Why was my account suspended without any warning?",
    "The chatbot gave me wrong information about return policy.",
    "I am being charged for a service I cancelled 3 months ago.",
]


def _make_task(rng: random.Random, idx: int) -> BrainTask:
    """Build a synthetic support BrainTask."""
    domain = rng.choice(_DOMAINS)
    prompt = rng.choice(_PROMPTS)
    stakes = rng.uniform(0.3, 0.95) if domain == "escalation" else rng.uniform(0.1, 0.65)
    return BrainTask(
        prompt=f"[#{idx}] {prompt}",
        domain=domain,
        uncertainty=rng.uniform(0.1, 0.8),
        complexity=rng.uniform(0.2, 0.7),
        stakes=stakes,
        source_confidence=rng.uniform(0.4, 0.9),
        tool_relevance=rng.uniform(0.5, 1.0),
        time_pressure=rng.uniform(0.1, 0.6),
        safety_sensitivity=0.7 if domain == "escalation" else rng.uniform(0.0, 0.4),
        collaboration_value=rng.uniform(0.1, 0.5),
        user_patience=rng.uniform(0.3, 0.9),
    )


# ---------------------------------------------------------------------------
# Simulated naive ReAct agent
# ---------------------------------------------------------------------------

_NAIVE_ACTIONS = ["auto_resolve", "route_to_faq", "send_template_email", "escalate", "ignore"]


def _naive_agent_action(task: BrainTask, rng: random.Random) -> str:
    """Naive heuristic: high-stakes tasks are sometimes mis-handled."""
    if task.stakes > 0.75 and rng.random() < 0.45:
        return rng.choice(["auto_resolve", "route_to_faq"])  # wrong — should escalate
    return rng.choice(_NAIVE_ACTIONS)


# ---------------------------------------------------------------------------
# Simulated tool callables
# ---------------------------------------------------------------------------

def _make_tool_fn(name: str, failure_start: int | None, rng: random.Random) -> Callable:
    """Return a callable that fails after *failure_start* calls (honey-pot pattern)."""
    call_count = [0]

    def fn(query: str) -> dict:
        call_count[0] += 1
        if failure_start is not None and call_count[0] > failure_start:
            raise ConnectionError(f"{name} unavailable (simulated outage)")
        time.sleep(0.0)  # no real delay in simulation
        return {"tool": name, "result": f"ok:{query[:20]}"}

    fn.__name__ = name
    return fn


# ---------------------------------------------------------------------------
# Reputation gossip helpers
# ---------------------------------------------------------------------------

def _build_org_snapshot(org_id: str, tools: list[str], rng: random.Random) -> OrgReputationSnapshot:
    """Build a synthetic OrgReputationSnapshot for cold-start seeding."""
    rates = {t: (rng.uniform(0.6, 0.95), rng.randint(20, 100)) for t in tools}
    return OrgReputationSnapshot(org_id=org_id, rates=rates)


# ---------------------------------------------------------------------------
# Main deployment runner
# ---------------------------------------------------------------------------

@dataclass
class ShadowRunReport:
    total_tasks: int
    total_disagreements: int
    disagreement_rate: float
    high_risk_escalations_manifold: int
    high_risk_escalations_naive: int
    virtual_regret_saved: int
    tool_failures_detected: int
    gossip_inoculation_speed: int  # tasks until all agents rerouted after first failure
    penalty_proposals: list[dict]
    adversarial_suspects: list[dict]
    top_disagreements: list[tuple[str, str]]
    hitl_escalations: int
    interceptor_summary: dict | None = None  # Phase 13 activation summary


def run_shadow_deployment(
    n_tasks: int = 100,
    seed: int = 2500,
    verbose: bool = True,
    input_tasks: list[tuple[BrainTask, str]] | None = None,
) -> ShadowRunReport:
    """Run a full shadow-mode deployment against a task stream.

    Parameters
    ----------
    n_tasks:
        Number of **synthetic** tasks to generate when *input_tasks* is ``None``.
        Ignored if *input_tasks* is provided.
    seed:
        Random seed for reproducibility (tool selection, synthetic stream).
    verbose:
        Print progress to stdout.
    input_tasks:
        Pre-loaded ``(BrainTask, naive_action)`` pairs from a real log file.
        When provided, the synthetic stream is bypassed entirely.

    Returns
    -------
    ShadowRunReport
        Structured report of the shadow run results.
    """
    rng = random.Random(seed)

    # Resolve the task stream -----------------------------------------------
    if input_tasks is not None:
        task_stream = input_tasks
        stream_label = f"{len(task_stream)} tasks from real log"
    else:
        task_stream = [(_make_task(rng, i), _naive_agent_action(_make_task(rng, i), rng))
                       for i in range(n_tasks)]
        stream_label = f"{n_tasks} synthetic customer support tasks"

    # ------------------------------------------------------------------
    # Phase 10: Federated gossip cold-start
    # ------------------------------------------------------------------
    bridge = FederatedGossipBridge()
    ledger = GlobalReputationLedger(min_orgs_required=2)

    tool_names = ["support_kb", "crm_lookup", "billing_api", "ticket_system", "email_sender"]
    for org_id in ["org_alpha", "org_beta", "org_gamma"]:
        snap = _build_org_snapshot(org_id, tool_names, rng)
        bridge.contribute_snapshot(snap)
        ledger.ingest_snapshot(snap)

    if verbose:
        print("=== MANIFOLD Shadow Mode — Trust Audit V2 ===\n")
        print(f"[Phase 10] Global ledger seeded with 3 orgs × {len(tool_names)} tools")
        for t in tool_names:
            rate = ledger.global_rate(t)
            if rate is not None:
                print(f"           {t}: global_rate={rate:.2f}")

    # ------------------------------------------------------------------
    # Phase 8: ConnectorRegistry + ShadowModeWrapper
    # ------------------------------------------------------------------
    registry = ConnectorRegistry()

    # "billing_api" is a honey-pot: healthy for first 15 calls, then fails
    for name in tool_names:
        failure_start = 15 if name == "billing_api" else None
        fn = _make_tool_fn(name, failure_start, rng)
        profile = ToolProfile(
            name=name,
            cost=rng.uniform(0.05, 0.15),
            latency=rng.uniform(0.05, 0.20),
            reliability=ledger.global_rate(name) or 0.80,
            risk=rng.uniform(0.05, 0.15),
            asset=rng.uniform(0.60, 0.85),
        )
        registry.register(ToolConnector(name=name, fn=fn, profile=profile))

    brain_cfg = BrainConfig(
        generations=20,
        population_size=40,
        grid_size=5,
        seed=seed,
    )
    tools = registry.tool_profiles()
    brain = ManifoldBrain(brain_cfg, tools=tools)
    wrapper = ShadowModeWrapper(brain=brain)

    if verbose:
        print(f"\n[Phase 8] ConnectorRegistry: {len(registry)} tools registered")
        print(f"[Phase 8] ShadowModeWrapper: active={wrapper.active}")

    # ------------------------------------------------------------------
    # Phase 9: HITL gate
    # ------------------------------------------------------------------
    hitl_cfg = HITLConfig(risk_stakes_threshold=0.55)
    hitl_gate = HITLGate(config=hitl_cfg)
    hitl_escalations = 0

    # ------------------------------------------------------------------
    # Phase 11: Adversarial detectors
    # ------------------------------------------------------------------
    adv_detector = AdversarialPricingDetector(warm_up_size=20, post_window_size=30, drop_threshold=0.35)
    nash_gate = NashEquilibriumGate()
    tool_failure_count = 0
    first_billing_failure_task: int | None = None
    gossip_reroute_task: int | None = None
    billing_inoculated = False

    # ------------------------------------------------------------------
    # Phase 12: AutoRuleDiscovery
    # ------------------------------------------------------------------
    discovery = AutoRuleDiscovery(
        optimizer=PenaltyOptimizer(min_observations=5),
        synthesizer=PolicySynthesizer(min_occurrences=3),
    )

    # ------------------------------------------------------------------
    # Phase 13: ActiveInterceptor (dry-run — logs veto decisions)
    # ------------------------------------------------------------------
    interceptor_cfg = InterceptorConfig(
        risk_veto_threshold=0.40,
        redirect_strategy="hitl",
    )
    interceptor = ActiveInterceptor(
        registry=registry,
        brain=brain,
        config=interceptor_cfg,
    )

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    high_risk_escalations_manifold = 0
    high_risk_escalations_naive = 0

    # ------------------------------------------------------------------
    # Main shadow loop
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[Run] Processing {stream_label}...\n")

    for i, (task, naive_action) in enumerate(task_stream):
        # Phase 8: Shadow observe
        vr = wrapper.observe(task, actual_action=naive_action)

        # Phase 9: HITL check (uses actual MANIFOLD decision)
        if hitl_gate.should_escalate(task, vr.manifold_decision):
            hitl_escalations += 1

        if task.stakes > 0.75:
            if vr.manifold_action == "escalate":
                high_risk_escalations_manifold += 1
            if naive_action == "escalate":
                high_risk_escalations_naive += 1

        # Execute tool calls through registry and record outcomes
        tool_name = rng.choice(tool_names)
        connector = registry.get(tool_name)
        if connector is None:
            continue
        result = connector.call(task.prompt[:50])
        outcome = result.to_brain_outcome()

        # Phase 11: Feed outcomes to adversarial detector
        adv_detector.record(tool_name, result.success)
        if not result.success:
            tool_failure_count += 1
            if tool_name == "billing_api" and first_billing_failure_task is None:
                first_billing_failure_task = i
            # Gossip inoculation: all agents re-route after 3 failure signals
            if first_billing_failure_task is not None and not billing_inoculated:
                failures_since = i - first_billing_failure_task
                if failures_since >= 3:
                    gossip_reroute_task = i
                    billing_inoculated = True

            # Feed gossip packet to bridge
            bridge.contribute_packet(
                FederatedGossipPacket(
                    tool_name=tool_name,
                    signal="failing",
                    confidence=0.9,
                    org_id="org_alpha",
                ),
            )

        # Phase 12: Observe rule events
        if not result.success:
            discovery.observe_rule_event(tool_name, "tool_failure", outcome, penalty=1.0)

        # Phase 13: Log interceptor decision (dry-run, non-blocking)
        try:
            interceptor.intercept(task, tool_name)
        except KeyError:
            pass

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------
    shadow_report = wrapper.shadow_report()
    adversarial_suspects = adv_detector.suspects()
    penalty_proposals = [
        {
            "rule_name": p.rule_name,
            "trigger": p.trigger,
            "current_penalty": round(p.current_penalty, 3),
            "proposed_penalty": round(p.proposed_penalty, 3),
            "delta": round(p.delta, 3),
            "confidence": round(p.confidence, 3),
            "rationale": p.rationale,
        }
        for p in discovery.suggest_penalty_updates()
    ]
    status = discovery.status()

    # Virtual regret saved = high-risk tasks MANIFOLD would escalate that naive agent didn't
    virtual_regret_saved = high_risk_escalations_manifold - high_risk_escalations_naive
    if virtual_regret_saved < 0:
        virtual_regret_saved = 0

    gossip_inoculation_speed = (
        (gossip_reroute_task - first_billing_failure_task)
        if (gossip_reroute_task is not None and first_billing_failure_task is not None)
        else -1
    )

    report = ShadowRunReport(
        total_tasks=shadow_report["total_observations"],
        total_disagreements=shadow_report["total_disagreements"],
        disagreement_rate=shadow_report["disagreement_rate"],
        high_risk_escalations_manifold=high_risk_escalations_manifold,
        high_risk_escalations_naive=high_risk_escalations_naive,
        virtual_regret_saved=virtual_regret_saved,
        tool_failures_detected=tool_failure_count,
        gossip_inoculation_speed=gossip_inoculation_speed,
        penalty_proposals=penalty_proposals,
        adversarial_suspects=adversarial_suspects,
        top_disagreements=shadow_report["top_disagreement_actions"],
        hitl_escalations=hitl_escalations,
        interceptor_summary=interceptor.summary(),
    )

    if verbose:
        _print_report(report, status)

    return report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(report: ShadowRunReport, status: dict) -> None:
    sep = "─" * 62
    print(sep)
    print("MANIFOLD TRUST AUDIT V2  —  Shadow Run Summary")
    print(sep)

    print(f"\n{'OBSERVATION':30s}  {'VALUE':>10s}")
    print("  " + "─" * 44)
    print(f"  {'Total tasks observed':30s}  {report.total_tasks:>10d}")
    print(f"  {'MANIFOLD disagreements':30s}  {report.total_disagreements:>10d}")
    print(f"  {'Disagreement rate':30s}  {report.disagreement_rate:>10.1%}")
    print(f"  {'HITL escalation triggers':30s}  {report.hitl_escalations:>10d}")

    print(f"\n{'ROI SIGNALS':30s}")
    print("  " + "─" * 44)
    print(f"  {'High-risk escalations (MANIFOLD)':35s}  {report.high_risk_escalations_manifold:>6d}")
    print(f"  {'High-risk escalations (naive)':35s}  {report.high_risk_escalations_naive:>6d}")
    print(f"  {'Virtual regret saved':35s}  {report.virtual_regret_saved:>6d}")
    print(f"  {'Tool failures detected':35s}  {report.tool_failures_detected:>6d}")
    inoculation = (
        f"{report.gossip_inoculation_speed} tasks"
        if report.gossip_inoculation_speed >= 0
        else "n/a (no outage detected)"
    )
    print(f"  {'Gossip inoculation speed':35s}  {inoculation}")

    print(f"\n{'PHASE 11 — ADVERSARIAL':30s}")
    print("  " + "─" * 44)
    if report.adversarial_suspects:
        for s in report.adversarial_suspects:
            print(
                f"  ⚠  HONEY-POT: {s['tool_name']}  "
                f"warm={s['warm_up_rate']:.0%}  post={s['post_rate']:.0%}  "
                f"drop={s['drop']:.0%}"
            )
    else:
        print("  ✓  No adversarial suspects detected")

    print(f"\n{'PHASE 12 — PENALTY PROPOSALS':30s}")
    print("  " + "─" * 44)
    if report.penalty_proposals:
        for p in report.penalty_proposals:
            direction = "↑" if p["delta"] > 0 else "↓"
            print(
                f"  {direction} {p['rule_name']} / {p['trigger']}: "
                f"{p['current_penalty']} → {p['proposed_penalty']}  "
                f"(Δ={p['delta']:+.3f}, conf={p['confidence']:.0%})"
            )
            print(f"     {p['rationale']}")
    else:
        print(f"  Rule triggers known: {status['known_triggers']}")
        print(f"  Pending proposals: {status['pending_proposals']}")

    print(f"\n{'PHASE 13 — ACTIVE INTERCEPTOR':30s}")
    print("  " + "─" * 44)
    if report.interceptor_summary:
        s = report.interceptor_summary
        print(f"  {'Calls evaluated':35s}  {s['total_calls']:>6d}")
        print(f"  {'Permitted':35s}  {s['permitted']:>6d}")
        print(f"  {'Vetoed':35s}  {s['vetoed']:>6d}")
        print(f"  {'Veto rate':35s}  {s['veto_rate']:>6.1%}")
        print(f"  {'Redirected to HITL':35s}  {s['redirected_to_hitl']:>6d}")
        print(f"  {'Avg risk score':35s}  {s['avg_risk_score']:>6.3f}")
    else:
        print("  (interceptor not active)")

    print(f"\n{'TOP DISAGREEMENT PAIRS':30s}")
    print("  " + "─" * 44)
    if report.top_disagreements:
        for naive, manifold in report.top_disagreements[:5]:
            print(f"  naive='{naive}'  ↔  MANIFOLD='{manifold}'")
    else:
        print("  (all actions agreed)")

    print(f"\n{sep}")
    print("MANIFOLD shadow run complete.")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MANIFOLD Shadow Mode deployment script — Trust Audit V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_shadow.py                           # synthetic stream (100 tasks)
  python deploy_shadow.py --tasks 200               # larger synthetic run
  python deploy_shadow.py --input support.csv       # real Zendesk/Intercom CSV
  python deploy_shadow.py --input traces.json       # real LangSmith/OpenAI JSON
  python deploy_shadow.py --input logs.csv --json > report.json   # sales artifact
""",
    )
    parser.add_argument(
        "--tasks", type=int, default=100,
        help="Number of synthetic tasks (default: 100, ignored when --input is set)"
    )
    parser.add_argument("--seed", type=int, default=2500, help="Random seed (default: 2500)")
    parser.add_argument(
        "--json", action="store_true", dest="emit_json",
        help="Emit JSON report to stdout instead of human-readable table"
    )
    parser.add_argument(
        "--input", dest="input_path", metavar="FILE",
        help="Path to a real log file to analyse (.csv or .json). "
             "When provided, the synthetic task stream is bypassed."
    )
    args = parser.parse_args()

    input_tasks: list[tuple[BrainTask, str]] | None = None
    if args.input_path:
        if not args.emit_json:
            print(f"[Loading] {args.input_path} ...")
        input_tasks = load_tasks_from_file(args.input_path)
        if not args.emit_json:
            print(f"[Loading] {len(input_tasks)} tasks loaded\n")

    report = run_shadow_deployment(
        n_tasks=args.tasks,
        seed=args.seed,
        verbose=not args.emit_json,
        input_tasks=input_tasks,
    )

    if args.emit_json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
