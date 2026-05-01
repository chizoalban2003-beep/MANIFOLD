"""MANIFOLD simulation package."""

from .ontogeny import (
    ENERGY_MAX,
    GenerationStats,
    LifetimeResult,
    StepDecision,
    Route,
    Vector,
    VectorGenome,
    armor_boost_for,
    clone_with_energy,
    default_routes,
    evolve,
    run_experiment,
    seed_population,
    summarize,
)

__all__ = [
    "ENERGY_MAX",
    "GenerationStats",
    "LifetimeResult",
    "StepDecision",
    "Route",
    "Vector",
    "VectorGenome",
    "armor_boost_for",
    "clone_with_energy",
    "default_routes",
    "evolve",
    "run_experiment",
    "seed_population",
    "summarize",
]
