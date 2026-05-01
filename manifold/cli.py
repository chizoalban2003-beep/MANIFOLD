"""Command-line runner for MANIFOLD experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .simulation import ManifoldConfig, run_manifold, summarize_result


def _build_config(args: argparse.Namespace) -> ManifoldConfig:
    config = ManifoldConfig(
        seed=args.seed,
        connector_events_path=str(args.connector_events) if args.connector_events else None,
    )
    if args.rulebook:
        config = replace(config, rulebook_text=args.rulebook.read_text(encoding="utf-8"))
    if args.quick:
        quick_phases = tuple(
            replace(phase, generations=max(4, phase.generations // 4))
            for phase in config.phases
        )
        config = replace(
            config,
            phases=quick_phases,
            initial_population=min(config.initial_population, 24),
            min_population=min(config.min_population, 16),
            max_population=min(config.max_population, 24),
        )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manifold",
        description="Run Project MANIFOLD evolutionary simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller version for fast smoke tests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for full simulation JSON.",
    )
    parser.add_argument(
        "--rulebook",
        type=Path,
        default=None,
        help="Optional DSL file for adaptive rule penalties.",
    )
    parser.add_argument(
        "--connector-events",
        type=Path,
        default=None,
        help="Optional CSV for live connector events.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _build_config(args)
    result = run_manifold(config=config)
    summary = summarize_result(result=result)

    print("=== MANIFOLD SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(result.to_json(indent=2), encoding="utf-8")
        print(f"\nFull telemetry saved to: {args.output_json}")


if __name__ == "__main__":
    main()
