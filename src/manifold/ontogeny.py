"""Phase 5 MANIFOLD simulation: ontogenetic energy budgeting.

The substrate stays intentionally compact. The point of Phase 5 is that each
vector now has to learn a lifetime policy: spend a finite battery on temporary
armor, or conserve energy and select a different route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import math
import random
from statistics import mean, pstdev
from typing import Callable, Iterable, Sequence


RiskSchedule = Callable[[int], float]
ENERGY_MAX = 30.0


@dataclass(frozen=True)
class Route:
    """A candidate path through the grid manifold."""

    name: str
    length: int
    base_cost: float
    risk_schedule: RiskSchedule

    def risk_at(self, generation: int) -> float:
        return self.risk_schedule(generation)


@dataclass(frozen=True)
class VectorGenome:
    """Inherited vector physics plus its lifetime armor policy."""

    risk_multiplier: float
    max_risk: float
    armor_threshold: float
    armor_efficiency: float
    conserve_bias: float

    @property
    def niche(self) -> str:
        if self.max_risk < 4.0:
            return "scout"
        if self.max_risk > 7.0:
            return "tank"
        return "hybrid"


@dataclass(frozen=True)
class StepDecision:
    timestep: int
    risk: float
    armor_delta: float
    energy_spent: float
    energy_remaining: float
    effective_risk: float


@dataclass(frozen=True)
class LifetimeResult:
    route_name: str
    survived: bool
    total_cost: float
    base_cost: float
    risk_cost: float
    energy_spent: float
    energy_remaining: float
    cognitive_load: float
    steps: tuple[StepDecision, ...]


@dataclass(frozen=True)
class Vector:
    """A living vector with a finite armor battery."""

    genome: VectorGenome
    energy_max: float = ENERGY_MAX

    def evaluate(self, routes: Sequence[Route], generation: int) -> LifetimeResult:
        """Select and traverse the lowest expected total-cost route."""

        if not routes:
            raise ValueError("at least one route is required")
        return min(
            (self.traverse(route, generation) for route in routes),
            key=lambda result: result.total_cost,
        )

    def traverse(self, route: Route, generation: int) -> LifetimeResult:
        """Apply the vector's finite-budget armor policy on one route."""

        if self.energy_max <= 0.0:
            raise ValueError("energy_max must be positive")

        risk = route.risk_at(generation)
        energy_remaining = self.energy_max
        risk_cost = 0.0
        energy_spent = 0.0
        policy_switches = 0
        previous_boosted = False
        steps: list[StepDecision] = []

        for timestep in range(route.length):
            armor_delta = armor_boost_for(
                risk=risk,
                genome=self.genome,
                energy_remaining=energy_remaining,
                timesteps_remaining=route.length - timestep,
            )
            step_energy = armor_delta / self.genome.armor_efficiency
            energy_remaining -= step_energy
            energy_spent += step_energy

            effective_risk = max(0.0, risk - armor_delta)
            risk_cost += effective_risk * self.genome.risk_multiplier
            boosted = armor_delta > 0.0
            if timestep > 0 and boosted != previous_boosted:
                policy_switches += 1
            previous_boosted = boosted

            steps.append(
                StepDecision(
                    timestep=timestep,
                    risk=risk,
                    armor_delta=armor_delta,
                    energy_spent=step_energy,
                    energy_remaining=energy_remaining,
                    effective_risk=effective_risk,
                )
            )

        survived = all(step.effective_risk <= self.genome.max_risk for step in steps)
        cognitive_load = energy_spent + policy_switches * 0.2
        survival_penalty = (
            0.0 if survived else 100.0 + max(0.0, risk - self.genome.max_risk) * 10.0
        )
        base_cost = route.base_cost + route.length
        total_cost = base_cost + risk_cost + energy_spent + survival_penalty

        return LifetimeResult(
            route_name=route.name,
            survived=survived,
            total_cost=total_cost,
            base_cost=base_cost,
            risk_cost=risk_cost,
            energy_spent=energy_spent,
            energy_remaining=energy_remaining,
            cognitive_load=cognitive_load,
            steps=tuple(steps),
        )


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    average_regret: float
    best_regret: float
    survival_rate: float
    diversity: float
    dominant_route: str
    niche_counts: dict[str, int]
    average_energy_spent: float
    average_cognitive_load: float


def armor_boost_for(
    risk: float,
    genome: VectorGenome,
    energy_remaining: float,
    timesteps_remaining: int,
) -> float:
    """Return per-step armor boost under finite battery constraints."""

    if risk <= genome.armor_threshold or energy_remaining <= 0.0:
        return 0.0

    desired_boost = max(0.0, risk - genome.max_risk)
    if desired_boost <= 0.0:
        return 0.0

    budget_fraction = max(0.05, min(1.0, 1.0 - genome.conserve_bias))
    per_step_energy_budget = energy_remaining / max(1, timesteps_remaining) * budget_fraction
    affordable_boost = per_step_energy_budget * genome.armor_efficiency
    return min(desired_boost, affordable_boost)


def flicker(low: float, high: float, period: int) -> RiskSchedule:
    """Build a route risk schedule that toggles between two regimes."""

    if period <= 0:
        raise ValueError("period must be positive")

    def schedule(generation: int) -> float:
        return low if (generation // period) % 2 == 0 else high

    return schedule


def default_routes() -> tuple[Route, ...]:
    """Routes matching the Phase 5 scaffold: tank, scout, and flicker corridor."""

    return (
        Route("tank", length=5, base_cost=1.5, risk_schedule=lambda _generation: 6.0),
        Route("scout", length=9, base_cost=0.5, risk_schedule=lambda _generation: 2.0),
        Route("flicker", length=6, base_cost=1.0, risk_schedule=flicker(3.0, 7.0, period=8)),
    )


def seed_population(size: int, rng: random.Random | None = None) -> list[Vector]:
    """Create a diverse generation-zero population across the physics space."""

    if size < 2:
        raise ValueError("population size must be at least 2")

    rng = rng or random.Random()
    population: list[Vector] = []
    for index in range(size):
        fraction = index / (size - 1)
        genome = VectorGenome(
            risk_multiplier=0.1 + fraction * 2.4,
            max_risk=2.0 + (1.0 - fraction) * 7.5,
            armor_threshold=rng.uniform(2.5, 7.0),
            armor_efficiency=rng.uniform(0.7, 1.4),
            conserve_bias=rng.uniform(0.1, 0.9),
        )
        population.append(Vector(genome=genome))
    rng.shuffle(population)
    return population


def evolve(
    population: Sequence[Vector],
    routes: Sequence[Route],
    generations: int,
    rng: random.Random | None = None,
    sigma: float = 0.05,
) -> tuple[list[Vector], list[GenerationStats]]:
    """Run selection with conservative mutation over ontogenetic rollouts."""

    if not population:
        raise ValueError("population cannot be empty")
    if not routes:
        raise ValueError("at least one route is required")
    if generations < 1:
        raise ValueError("generations must be positive")

    rng = rng or random.Random()
    current = list(population)
    history: list[GenerationStats] = []

    for generation in range(generations):
        evaluated = [(vector, vector.evaluate(routes, generation)) for vector in current]
        optimum = min(result.total_cost for _vector, result in evaluated)
        regrets = [result.total_cost - optimum for _vector, result in evaluated]
        history.append(_stats_for(generation, evaluated, regrets))

        ranked = sorted(
            zip(current, evaluated, regrets, strict=True),
            key=lambda item: item[2] + _niche_penalty(item[0], current),
        )
        survivor_count = max(2, len(current) // 2)
        survivors = [item[0] for item in ranked[:survivor_count]]

        next_generation = survivors.copy()
        while len(next_generation) < len(current):
            parent = rng.choice(survivors)
            next_generation.append(mutate(parent, rng, sigma=sigma))
        current = next_generation

    return current, history


def mutate(vector: Vector, rng: random.Random | None = None, sigma: float = 0.05) -> Vector:
    """Conservatively mutate a vector around its parent policy."""

    rng = rng or random.Random()
    genome = vector.genome
    return Vector(
        genome=VectorGenome(
            risk_multiplier=_clamp(genome.risk_multiplier + rng.gauss(0, sigma), 0.1, 2.5),
            max_risk=_clamp(genome.max_risk + rng.gauss(0, sigma * 3.0), 2.0, 9.5),
            armor_threshold=_clamp(genome.armor_threshold + rng.gauss(0, sigma * 4.0), 2.0, 8.0),
            armor_efficiency=_clamp(genome.armor_efficiency + rng.gauss(0, sigma), 0.5, 1.8),
            conserve_bias=_clamp(genome.conserve_bias + rng.gauss(0, sigma), 0.0, 0.95),
        ),
        energy_max=vector.energy_max,
    )


def run_experiment(
    generations: int = 40,
    population_size: int = 32,
    seed: int = 7,
) -> tuple[list[Vector], list[GenerationStats]]:
    """Convenience entry point for notebooks and CLI use."""

    rng = random.Random(seed)
    population = seed_population(population_size, rng)
    return evolve(population, default_routes(), generations, rng)


def summarize(history: Iterable[GenerationStats]) -> str:
    """Format generation metrics for CLI or notebook display."""

    lines = [
        "generation,average_regret,best_regret,survival_rate,diversity,"
        "dominant_route,niche_counts,average_energy_spent,average_cognitive_load"
    ]
    for stats in history:
        lines.append(
            f"{stats.generation},{stats.average_regret:.3f},{stats.best_regret:.3f},"
            f"{stats.survival_rate:.3f},{stats.diversity:.3f},{stats.dominant_route},"
            f"{stats.niche_counts},{stats.average_energy_spent:.3f},"
            f"{stats.average_cognitive_load:.3f}"
        )
    return "\n".join(lines)


def _stats_for(
    generation: int,
    evaluated: Sequence[tuple[Vector, LifetimeResult]],
    regrets: Sequence[float],
) -> GenerationStats:
    route_counts: dict[str, int] = {}
    niche_counts = {"scout": 0, "hybrid": 0, "tank": 0}
    for vector, result in evaluated:
        route_counts[result.route_name] = route_counts.get(result.route_name, 0) + 1
        niche_counts[vector.genome.niche] += 1

    return GenerationStats(
        generation=generation,
        average_regret=mean(regrets),
        best_regret=min(regrets),
        survival_rate=mean(1.0 if result.survived else 0.0 for _vector, result in evaluated),
        diversity=_population_diversity(vector for vector, _result in evaluated),
        dominant_route=max(route_counts, key=route_counts.get),
        niche_counts=niche_counts,
        average_energy_spent=mean(result.energy_spent for _vector, result in evaluated),
        average_cognitive_load=mean(result.cognitive_load for _vector, result in evaluated),
    )


def _population_diversity(vectors: Iterable[Vector]) -> float:
    genomes = [vector.genome for vector in vectors]
    if len(genomes) < 2:
        return 0.0

    spreads = [
        pstdev([genome.risk_multiplier for genome in genomes]) / 2.4,
        pstdev([genome.max_risk for genome in genomes]) / 7.5,
        pstdev([genome.armor_threshold for genome in genomes]) / 6.0,
        pstdev([genome.armor_efficiency for genome in genomes]) / 1.3,
        pstdev([genome.conserve_bias for genome in genomes]) / 0.95,
    ]
    return math.fsum(spreads)


def _niche_penalty(vector: Vector, population: Sequence[Vector]) -> float:
    neighbors = 0
    for other in population:
        if other is vector:
            continue
        distance = abs(vector.genome.max_risk - other.genome.max_risk) + abs(
            vector.genome.risk_multiplier - other.genome.risk_multiplier
        )
        if distance < 0.75:
            neighbors += 1
    return neighbors * 0.04


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def clone_with_energy(vector: Vector, energy_max: float) -> Vector:
    """Convenience helper for experiments that vary battery capacity."""

    if energy_max <= 0.0:
        raise ValueError("energy_max must be positive")
    return replace(vector, energy_max=energy_max)


def main() -> None:
    """Run a Phase 5 experiment from the command line."""

    parser = argparse.ArgumentParser(description="Run MANIFOLD Phase 5 ontogeny.")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    _final_population, history = run_experiment(
        generations=args.generations,
        population_size=args.population_size,
        seed=args.seed,
    )
    print(summarize(history))
