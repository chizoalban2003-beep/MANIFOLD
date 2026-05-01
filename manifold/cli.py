"""Command-line runner for Project MANIFOLD."""

from __future__ import annotations

import argparse
import json

from manifold.simulation import SimulationConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Project MANIFOLD evolutionary experiment."
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", "--population", type=int, default=52)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--grid-size", type=int, default=11)
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
    config = SimulationConfig(
        generations=args.generations,
        population_size=args.population_size,
        seed=args.seed,
        grid_size=args.grid_size,
        teacher_mode=args.teacher_mode,
        communication_enabled=args.communication,
        recharge_enabled=not args.no_recharge,
    )
    history = run_experiment(config)
    if args.json:
        print(json.dumps([summary.__dict__ for summary in history], indent=2))
        return

    final = history[-1]
    print("Project MANIFOLD")
    print(f"Generations: {len(history)}")
    print(f"Survival rate: {final.survival_rate:.2%}")
    print(f"Average regret: {final.average_regret:.2f}")
    print(f"Average energy spent: {final.average_energy_spent:.2f}")
    print(f"Average charger visits: {final.average_recharge_visits:.2f}")
    print(f"Average max_r: {final.average_max_risk:.2f}")
    print(f"Average aversion: {final.average_energy_aversion:.2f}")
    print(f"Diversity: {final.diversity:.2f}")
    print(f"Niches: {final.niche_counts}")


if __name__ == "__main__":
    main()
