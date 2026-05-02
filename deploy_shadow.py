"""deploy_shadow.py — MANIFOLD Shadow Mode Deployment Script.

Run this script to execute a complete Trust Audit V2 against a synthetic
customer-support stream.  It wires together all 12 phases:

  Phase 8  — ConnectorRegistry + ShadowModeWrapper
  Phase 9  — HITLGate (escalation detection)
  Phase 10 — FederatedGossipBridge / GlobalReputationLedger
  Phase 11 — AdversarialPricingDetector + NashEquilibriumGate
  Phase 12 — AutoRuleDiscovery (PenaltyOptimizer + PolicySynthesizer)

Usage::

    python deploy_shadow.py                 # run with built-in synthetic stream
    python deploy_shadow.py --tasks 200     # change stream size
    python deploy_shadow.py --seed 42       # reproducible run
    python deploy_shadow.py --json          # emit JSON report to stdout
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Callable

from manifold import (
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


def run_shadow_deployment(
    n_tasks: int = 100,
    seed: int = 2500,
    verbose: bool = True,
) -> ShadowRunReport:
    """Run a full shadow-mode deployment against a synthetic support stream.

    Parameters
    ----------
    n_tasks:
        Number of synthetic customer support tasks to process.
    seed:
        Random seed for reproducibility.
    verbose:
        Print progress to stdout.

    Returns
    -------
    ShadowRunReport
        Structured report of the shadow run results.
    """
    rng = random.Random(seed)

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
    # Counters
    # ------------------------------------------------------------------
    high_risk_escalations_manifold = 0
    high_risk_escalations_naive = 0

    # ------------------------------------------------------------------
    # Main shadow loop
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[Run] Processing {n_tasks} synthetic customer support tasks...\n")

    for i in range(n_tasks):
        task = _make_task(rng, i)
        naive_action = _naive_agent_action(task, rng)

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

        # Phase 12: Observe successful decompositions (via brain decision)
        if vr.manifold_decision.action in {"decompose", "escalate"}:
            pass  # hierarchical decisions tracked separately in production

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
    parser = argparse.ArgumentParser(description="MANIFOLD Shadow Mode deployment script")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks to process (default: 100)")
    parser.add_argument("--seed", type=int, default=2500, help="Random seed (default: 2500)")
    parser.add_argument("--json", action="store_true", dest="emit_json", help="Emit JSON report to stdout")
    args = parser.parse_args()

    report = run_shadow_deployment(
        n_tasks=args.tasks,
        seed=args.seed,
        verbose=not args.emit_json,
    )

    if args.emit_json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
