"""Command-line runner for Project MANIFOLD."""

from __future__ import annotations

import argparse
import json

from manifold.simulation import SimulationConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MANIFOLD Phase 5 ontogeny experiment."
    )
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=36)
    parser.add_argument("--seed", type=int, default=13)
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
        recharge_enabled=not args.no_recharge,
    )
    history = run_experiment(config)
    if args.json:
        print(json.dumps([summary.__dict__ for summary in history], indent=2))
        return

    final = history[-1]
    print("Project MANIFOLD - Phase 5 ontogeny")
    print(f"Generations: {len(history)}")
    print(f"Average regret: {final.average_regret:.2f}")
    print(f"Best regret: {final.best_regret:.2f}")
    print(f"Average energy spent: {final.average_energy_spent:.2f}")
    print(f"Diversity: {final.diversity:.2f}")
    print(f"Niches: {final.niche_counts}")


if __name__ == "__main__":
    main()
