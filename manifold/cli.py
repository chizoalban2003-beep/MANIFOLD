"""Command-line runner for Project MANIFOLD."""

from __future__ import annotations

import argparse
import json

from manifold.simulation import SimulationConfig, run_experiment
from manifold.social import (
    SocialConfig,
    compile_policy_audit,
    config_for_preset,
    run_social_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Project MANIFOLD evolutionary experiment."
    )
    parser.add_argument(
        "--mode",
        choices=["path", "social"],
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full generation history as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "path":
        history = run_path_mode(args)
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


if __name__ == "__main__":
    main()
