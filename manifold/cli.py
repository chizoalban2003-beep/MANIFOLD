"""Command-line runner for Project MANIFOLD."""

from __future__ import annotations

import argparse
import json

from manifold.brain import BrainConfig, BrainTask, ManifoldBrain, default_tools
from manifold.brainbench import load_brain_tasks_csv, run_brain_benchmark, sample_brain_tasks
from manifold.gridmapper import AgentPopulation, GridWorld
from manifold.research import format_research_report, run_research_suite
from manifold.simulation import SimulationConfig, run_experiment
from manifold.social import (
    SocialConfig,
    compile_policy_audit,
    config_for_preset,
    run_social_experiment,
)
from manifold.trustrouter import DialogueTask, TrustRouter, TrustRouterConfig
from manifold.trustbench import (
    load_labelled_tasks_csv,
    run_trust_benchmark,
    sample_trust_tasks,
)
from manifold.trustaudit import format_trust_audit_report, run_support_trust_audit


# ---------------------------------------------------------------------------
# Agent subcommand helpers
# ---------------------------------------------------------------------------

def _build_agent_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the 'agent' subcommand group."""
    agent_p = subparsers.add_parser("agent", help="Manage MANIFOLD agents.")
    agent_sub = agent_p.add_subparsers(dest="agent_cmd")

    add_p = agent_sub.add_parser(
        "add",
        help=(
            "Register a new agent from a pre-filled profile. "
            "Example: python -m manifold agent add claude-sonnet my-agent-01 'My Sonnet'"
        ),
    )
    add_p.add_argument("profile", nargs="?", default=None, help="Profile name (e.g. claude-sonnet, roomba, drone).")
    add_p.add_argument("agent_id", nargs="?", default=None, help="Unique agent ID to register.")
    add_p.add_argument("display_name", nargs="?", default=None, help="Human-readable agent name.")
    add_p.add_argument(
        "--url",
        default="http://localhost:8080",
        help="MANIFOLD server URL (default: http://localhost:8080).",
    )
    add_p.add_argument(
        "--org",
        default="default",
        dest="org_id",
        help="Organisation ID (default: default).",
    )
    add_p.add_argument(
        "--key",
        default="",
        dest="api_key",
        help="MANIFOLD API key (Bearer token).",
    )
    add_p.add_argument(
        "--endpoint",
        default="",
        help="Agent's own callback endpoint URL (optional).",
    )
    add_p.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all available profile names and exit.",
    )

    list_p = agent_sub.add_parser("list-profiles", help="List all available agent profiles.")
    list_p.add_argument("--json", action="store_true", help="Emit JSON output.")


def _run_agent_add(args: argparse.Namespace) -> None:
    """Execute 'agent add' — register an agent via the MANIFOLD server API."""
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    from manifold.agent_profiles import get_profile, list_profiles  # noqa: PLC0415

    if getattr(args, "list_profiles", False):
        print("Available agent profiles:")
        for name in list_profiles():
            print(f"  {name}")
        return

    if not args.profile or not args.agent_id or not args.display_name:
        print("Usage: python -m manifold agent add <profile> <agent_id> <display_name> [options]")
        print("       python -m manifold agent add --list-profiles")
        return

    try:
        profile = get_profile(args.profile)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    payload = {
        "agent_id": args.agent_id,
        "display_name": args.display_name,
        "capabilities": profile["capabilities"],
        "domain": profile["domain"],
        "org_id": args.org_id,
        "endpoint_url": args.endpoint,
    }

    base = args.url.rstrip("/")
    url = f"{base}/agents/register"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        print(f"✓ Agent registered: {result.get('agent_id', args.agent_id)}")
        print(f"  Display name : {result.get('display_name', args.display_name)}")
        print(f"  Domain       : {result.get('domain', profile['domain'])}")
        print(f"  Capabilities : {', '.join(payload['capabilities'])}")
        print(f"  Layer        : {profile['layer']}  (level {profile['level']})")
        crna = profile["crna_profile"]
        print(
            f"  CRNA profile : C={crna['c']}  R={crna['r']}  N={crna['n']}  A={crna['a']}"
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"✗ Registration failed (HTTP {exc.code}): {body}")
    except OSError as exc:
        print(f"✗ Could not reach server at {url}: {exc}")
        print("  Hint: start the server with: python -m manifold.server --port 8080")


def _run_agent_list_profiles(args: argparse.Namespace) -> None:
    """Execute 'agent list-profiles'."""
    from manifold.agent_profiles import AGENT_PROFILES  # noqa: PLC0415

    if getattr(args, "json", False):
        print(json.dumps(AGENT_PROFILES, indent=2))
        return
    print("Available agent profiles:")
    for name, prof in sorted(AGENT_PROFILES.items()):
        caps = ", ".join(prof["capabilities"])
        print(f"  {name:<20} level={prof['level']}  {prof['layer']:<22} [{caps}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Project MANIFOLD evolutionary experiment."
    )
    # Top-level subcommands (e.g. 'agent')
    subparsers = parser.add_subparsers(dest="subcommand")
    _build_agent_parser(subparsers)

    parser.add_argument(
        "--mode",
        choices=[
            "path",
            "social",
            "gridmapper",
            "trustrouter",
            "trustbench",
            "brain",
            "brainbench",
            "research",
            "trustaudit",
        ],
        default="social",
        help="Run the path/teacher engine or the social-intelligence engine.",
    )
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--population-size", "--population", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2500)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--data-path",
        help="Optional CSV grid with row,col,cost,risk,asset[,neutrality] columns.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="GridMapper target as id,row,col,asset[,moves]. May be supplied multiple times.",
    )
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        help="GridMapper rule as name,penalty,trigger. Triggers include miss_target, deception_detected, trusted_lie, low_energy.",
    )
    parser.add_argument(
        "--preset",
        choices=["trust", "birmingham", "misinformation", "compute"],
        default="trust",
        help="Social mapper preset.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["periodic", "reactive", "random", "adversarial", "multi"],
        default="periodic",
    )
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--no-recharge", action="store_true")
    parser.add_argument("--prompt", default="How should I answer this user?")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--uncertainty", type=float, default=0.5)
    parser.add_argument("--complexity", type=float, default=0.5)
    parser.add_argument("--stakes", type=float, default=0.5)
    parser.add_argument("--source-confidence", type=float, default=0.7)
    parser.add_argument("--user-patience", type=float, default=0.7)
    parser.add_argument("--safety-sensitivity", type=float, default=0.2)
    parser.add_argument("--dynamic-intent", action="store_true")
    parser.add_argument("--tool-relevance", type=float, default=0.5)
    parser.add_argument("--time-pressure", type=float, default=0.4)
    parser.add_argument("--collaboration-value", type=float, default=0.3)
    parser.add_argument(
        "--tasks-path",
        help="Optional TrustBench CSV with prompt,expected_action and task feature columns.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full generation history as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Route agent subcommand before legacy --mode dispatch
    if getattr(args, "subcommand", None) == "agent":
        agent_cmd = getattr(args, "agent_cmd", None)
        if agent_cmd == "add":
            _run_agent_add(args)
        elif agent_cmd == "list-profiles":
            _run_agent_list_profiles(args)
        else:
            print("Usage: python -m manifold agent {add,list-profiles} ...")
        return

    if args.mode == "path":
        history = run_path_mode(args)
    elif args.mode == "gridmapper":
        history = run_gridmapper_mode(args)
    elif args.mode == "trustrouter":
        history = run_trustrouter_mode(args)
    elif args.mode == "trustbench":
        history = run_trustbench_mode(args)
    elif args.mode == "brain":
        history = run_brain_mode(args)
    elif args.mode == "brainbench":
        history = run_brainbench_mode(args)
    elif args.mode == "research":
        history = run_research_mode(args)
    elif args.mode == "trustaudit":
        history = run_trustaudit_mode(args)
    else:
        history = run_social_mode(args)

    if args.json:
        print(json.dumps([summary.__dict__ for summary in history], indent=2))


def run_path_mode(args: argparse.Namespace):
    config = SimulationConfig(
        generations=args.generations,
        population_size=args.population_size,
        seed=args.seed,
        grid_size=args.grid_size if args.grid_size in (11, 21, 31) else 11,
        teacher_mode=args.teacher_mode,
        communication_enabled=args.communication,
        recharge_enabled=not args.no_recharge,
    )
    history = run_experiment(config)
    if not args.json:
        final = history[-1]
        print("Project MANIFOLD - path/teacher engine")
        print(f"Generations: {len(history)}")
        print(f"Survival rate: {final.survival_rate:.2%}")
        print(f"Average regret: {final.average_regret:.2f}")
        print(f"Average energy spent: {final.average_energy_spent:.2f}")
        print(f"Average charger visits: {final.average_recharge_visits:.2f}")
        print(f"Average max_r: {final.average_max_risk:.2f}")
        print(f"Average aversion: {final.average_energy_aversion:.2f}")
        print(f"Diversity: {final.diversity:.2f}")
        print(f"Niches: {final.niche_counts}")
    return history


def run_social_mode(args: argparse.Namespace):
    if args.preset == "trust":
        config = SocialConfig(
            generations=args.generations,
            population_size=args.population_size,
            seed=args.seed,
            grid_size=args.grid_size,
            preset=args.preset,
            data_path=args.data_path,
        )
    else:
        config = config_for_preset(
            args.preset,
            generations=args.generations,
            population_size=args.population_size,
            seed=args.seed,
        )
        if args.data_path:
            config = SocialConfig(
                generations=config.generations,
                population_size=config.population_size,
                seed=config.seed,
                grid_size=args.grid_size,
                preset=config.preset,
                signal_cost=config.signal_cost,
                verification_cost=config.verification_cost,
                false_trust_penalty=config.false_trust_penalty,
                detected_lie_penalty=config.detected_lie_penalty,
                data_path=args.data_path,
            )
    history = run_social_experiment(config)
    if not args.json:
        final = history[-1]
        audit = compile_policy_audit(history, config)
        print("Project MANIFOLD - social intelligence engine")
        print(f"Preset: {args.preset}")
        print(f"Generations: {len(history)}")
        print(f"Average fitness: {final.average_fitness:.2f}")
        print(f"Best fitness: {final.best_fitness:.2f}")
        print(f"Deception: {final.average_deception:.2%}")
        print(f"Verification: {final.average_verification:.2%}")
        print(f"Gossip: {final.average_gossip:.2%}")
        print(f"Memory: {final.average_memory_ticks:.1f} ticks")
        print(f"Lie rate: {final.lie_rate:.2%}")
        print(f"Blacklist rate: {final.blacklist_rate:.2f}")
        print(f"Forgiveness rate: {final.forgiveness_rate:.2f}")
        print(f"Predatory scout checks: {final.predatory_scout_rate:.2%}")
        print(f"Top source share: {final.top_source_share:.2%}")
        print(f"Monopoly risk: {audit.monopoly_risk:.2%}")
        print(f"Robustness score: {audit.robustness_score:.2f}")
        print("Policy recommendations:")
        print(f"  Verify above lie probability: {audit.verification_threshold:.2%}")
        print(f"  Target verification rate: {audit.recommended_verification_rate:.2%}")
        print(f"  Target gossip rate: {audit.recommended_gossip_rate:.2%}")
        print(f"  Predation threshold: {audit.recommended_predation_threshold:.2%}")
        print(f"  Blacklist after lies: {audit.recommended_blacklist_after_lies}")
        print(f"  Forgiveness window: {audit.recommended_forgiveness_window} ticks")
        print("  Monopoly controls: " + ", ".join(audit.monopoly_controls))
        print(f"Niches: {final.niche_counts}")
    return history


def run_gridmapper_mode(args: argparse.Namespace):
    world = GridWorld(size=args.grid_size)
    if args.data_path:
        world.load_from_csv(args.data_path)
    for target in args.target:
        target_id, row, col, asset, *rest = target.split(",")
        world.add_dynamic_targets(
            [
                {
                    "id": target_id,
                    "pos": (int(row), int(col)),
                    "asset": float(asset),
                    "moves": rest[0] if rest else "static",
                }
            ]
        )
    for rule in args.rule:
        name, penalty, trigger = rule.split(",")
        world.add_rule(name, float(penalty), trigger)
    if not world.targets:
        center = args.grid_size // 2
        world.add_dynamic_targets(
            [{"id": "default_target", "pos": (center, center), "asset": 1.0}]
        )

    result = AgentPopulation(seed=str(args.seed), n=args.population_size).optimize(
        world,
        generations=args.generations,
    )
    if not args.json:
        final = result.history[-1]
        print("Project MANIFOLD - GridMapper OS")
        print(f"Generations: {len(result.history)}")
        print(f"Targets: {result.target_snapshots.get(len(result.history) - 1, ())}")
        print(f"Average fitness: {final.average_fitness:.2f}")
        print(f"Verification: {result.verification:.2%}")
        print(f"Gossip: {result.gossip:.2%}")
        print(f"Reputation cap: {result.reputation_cap:.2%}")
        print(f"Rule penalty budget: {result.rule_penalty_budget:.2f}")
        print(f"Robustness score: {result.audit.robustness_score:.2f}")
        print(f"Niches: {final.niche_counts}")
    return result.history


def run_trustrouter_mode(args: argparse.Namespace):
    router = TrustRouter(
        TrustRouterConfig(
            generations=args.generations,
            population_size=args.population_size,
            grid_size=args.grid_size if args.grid_size in (5, 11, 21, 31) else 11,
            seed=args.seed,
        )
    )
    task = DialogueTask(
        prompt=args.prompt,
        domain=args.domain,
        uncertainty=args.uncertainty,
        complexity=args.complexity,
        stakes=args.stakes,
        source_confidence=args.source_confidence,
        user_patience=args.user_patience,
        safety_sensitivity=args.safety_sensitivity,
        dynamic_intent=args.dynamic_intent,
    )
    decision = router.route(task)
    if not args.json:
        print("Project MANIFOLD - TrustRouter")
        print(f"Action: {decision.action}")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"Risk score: {decision.risk_score:.2%}")
        print(f"Verification threshold: {decision.verification_threshold:.2%}")
        print(f"Clarification threshold: {decision.clarification_threshold:.2%}")
        print(f"Retrieval threshold: {decision.retrieval_threshold:.2%}")
        print(f"Escalation threshold: {decision.escalation_threshold:.2%}")
        print(f"Recommended verification: {decision.recommended_verification_rate:.2%}")
        print(f"Recommended gossip: {decision.recommended_gossip_rate:.2%}")
        print(f"Reputation cap: {decision.reputation_cap:.2%}")
        print(f"Robustness score: {decision.robustness_score:.2f}")
        print("Notes:")
        for note in decision.notes:
            print(f"  - {note}")
    return decision.result.history


def run_trustbench_mode(args: argparse.Namespace):
    tasks = load_labelled_tasks_csv(args.tasks_path) if args.tasks_path else sample_trust_tasks()
    report = run_trust_benchmark(
        tasks,
        TrustRouterConfig(
            generations=args.generations,
            population_size=args.population_size,
            grid_size=args.grid_size if args.grid_size in (5, 11, 21, 31) else 11,
            seed=args.seed,
        ),
    )
    if not args.json:
        print("Project MANIFOLD - TrustBench")
        print(f"Tasks: {len(tasks)}")
        print(f"Best policy: {report.best_policy}")
        print(f"TrustRouter rank: {report.trustrouter_rank}")
        for score in sorted(report.scores, key=lambda item: item.utility, reverse=True):
            print(
                f"{score.name}: utility={score.utility:.3f}, "
                f"accuracy={score.accuracy:.2%}, "
                f"cost={score.average_action_cost:.3f}, "
                f"risk_penalty={score.average_risk_penalty:.3f}, "
                f"missed={score.missed_verification_rate:.2%}, "
                f"unnecessary={score.unnecessary_verification_rate:.2%}"
            )
        print("Recommendations:")
        for recommendation in report.recommendations:
            print(f"  - {recommendation}")
    return []


def run_brain_mode(args: argparse.Namespace):
    brain = ManifoldBrain(
        BrainConfig(
            generations=args.generations,
            population_size=args.population_size,
            grid_size=args.grid_size if args.grid_size in (5, 11, 21, 31) else 11,
            seed=args.seed,
        ),
        default_tools(),
    )
    task = BrainTask(
        prompt=args.prompt,
        domain=args.domain,
        uncertainty=args.uncertainty,
        complexity=args.complexity,
        stakes=args.stakes,
        source_confidence=args.source_confidence,
        tool_relevance=args.tool_relevance,
        time_pressure=args.time_pressure,
        safety_sensitivity=args.safety_sensitivity,
        collaboration_value=args.collaboration_value,
        user_patience=args.user_patience,
        dynamic_goal=args.dynamic_intent,
    )
    decision = brain.decide(task)
    if not args.json:
        print("Project MANIFOLD - Brain")
        print(f"Action: {decision.action}")
        print(f"Selected tool: {decision.selected_tool or 'none'}")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"Risk score: {decision.risk_score:.2%}")
        print(f"Expected utility: {decision.expected_utility:.3f}")
        print(f"Verification: {decision.verification_rate:.2%}")
        print(f"Reputation cap: {decision.reputation_cap:.2%}")
        print(f"Robustness score: {decision.robustness_score:.2f}")
        print("Notes:")
        for note in decision.notes:
            print(f"  - {note}")
    return decision.result.history


def run_brainbench_mode(args: argparse.Namespace):
    tasks = load_brain_tasks_csv(args.tasks_path) if args.tasks_path else sample_brain_tasks()
    report = run_brain_benchmark(
        tasks,
        BrainConfig(
            generations=args.generations,
            population_size=args.population_size,
            grid_size=args.grid_size if args.grid_size in (5, 11, 21, 31) else 11,
            seed=args.seed,
        ),
    )
    if not args.json:
        print("Project MANIFOLD - BrainBench")
        print(f"Tasks: {len(tasks)}")
        print(f"Best policy: {report.best_policy}")
        print(f"MANIFOLD Brain rank: {report.brain_rank}")
        for score in sorted(report.scores, key=lambda item: item.utility, reverse=True):
            print(
                f"{score.name}: utility={score.utility:.3f}, "
                f"accuracy={score.accuracy:.2%}, "
                f"cost={score.average_action_cost:.3f}, "
                f"risk_penalty={score.average_risk_penalty:.3f}, "
                f"missed_safety={score.missed_safety_rate:.2%}, "
                f"over_tool={score.over_tool_rate:.2%}"
            )
        print("Recommendations:")
        for recommendation in report.recommendations:
            print(f"  - {recommendation}")
    return []


def run_research_mode(args: argparse.Namespace):
    report = run_research_suite(seed=args.seed)
    if not args.json:
        print(format_research_report(report))
    return []


def run_trustaudit_mode(args: argparse.Namespace):
    report = run_support_trust_audit()
    if not args.json:
        print(format_trust_audit_report(report))
    return []


if __name__ == "__main__":
    main()
