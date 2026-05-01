"""Core MANIFOLD simulation engine.

The model intentionally keeps the grid small so the experiment remains legible:
agents differ by physics, evaluate routes through non-stationary risk fields, and
now carry an energy battery that can be spent on temporary armor during a life.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from statistics import fmean
from typing import Iterable


Position = tuple[int, int]


GRID_SIZE = 5
START: Position = (0, 2)
GOAL: Position = (4, 2)
ENERGY_MAX = 30.0


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a MANIFOLD experiment."""

    population_size: int = 36
    generations: int = 60
    seed: int = 13
    teacher_interval: int = 15
    flicker_period: int = 8
    mutation_sigma: float = 0.05
    targeted_teacher_rate: float = 0.70
    recharge_enabled: bool = True
    recharge_amount: float = 12.0
    recharge_cells: tuple[Position, ...] = ((2, 0), (2, 4))
    energy_max: float = ENERGY_MAX
    tournament_survival_rate: float = 0.50
    diversity_penalty: float = 0.06


@dataclass
class VectorGenome:
    """Heritable vector physics plus ontogenetic energy-use tendencies."""

    risk_multiplier: float
    max_risk: float
    armor_bias: float
    conserve_bias: float
    recharge_bias: float
    ancestry: str = "seed"

    def niche(self) -> str:
        if self.max_risk >= 7.2:
            return "Tank"
        if self.max_risk <= 4.0 and self.risk_multiplier <= 0.9:
            return "Scout"
        return "Hybrid"

    def mutated(self, rng: random.Random, sigma: float) -> "VectorGenome":
        def clamp(value: float, low: float, high: float) -> float:
            return max(low, min(high, value))

        return VectorGenome(
            risk_multiplier=clamp(
                self.risk_multiplier + rng.gauss(0.0, sigma * 2.4), 0.1, 2.5
            ),
            max_risk=clamp(self.max_risk + rng.gauss(0.0, sigma * 8.0), 2.0, 9.5),
            armor_bias=clamp(self.armor_bias + rng.gauss(0.0, sigma * 1.8), 0.0, 1.0),
            conserve_bias=clamp(
                self.conserve_bias + rng.gauss(0.0, sigma * 1.8), 0.0, 1.0
            ),
            recharge_bias=clamp(
                self.recharge_bias + rng.gauss(0.0, sigma * 1.8), 0.0, 1.0
            ),
            ancestry=self.niche(),
        )


@dataclass(frozen=True)
class LifeResult:
    """One vector's route, costs, and energy decisions for a generation."""

    genome: VectorGenome
    generation: int
    route: tuple[Position, ...]
    reached_goal: bool
    base_cost: float
    energy_spent: float
    energy_remaining: float
    recharge_visits: int
    regret: float
    fitness: float
    niche: str


@dataclass(frozen=True)
class GenerationSummary:
    """Population-level metrics emitted after each generation."""

    generation: int
    average_regret: float
    best_regret: float
    average_energy_spent: float
    average_energy_remaining: float
    diversity: float
    teacher_mutated: bool
    flicker_risk: float
    niche_counts: dict[str, int]


@dataclass
class ManifoldExperiment:
    """Runnable experiment state."""

    config: SimulationConfig = field(default_factory=SimulationConfig)
    rng: random.Random = field(init=False)
    population: list[VectorGenome] = field(init=False)
    pheromone: dict[Position, float] = field(default_factory=dict)
    teacher_spikes: dict[Position, float] = field(default_factory=dict)
    history: list[GenerationSummary] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.population = seed_population(self.config.population_size, self.rng)

    def run(self) -> list[GenerationSummary]:
        for generation in range(self.config.generations):
            results, teacher_mutated = self.step(generation)
            self.history.append(self._summarize(generation, results, teacher_mutated))
        return self.history

    def step(self, generation: int) -> tuple[list[LifeResult], bool]:
        results = [self.evaluate_vector(genome, generation) for genome in self.population]
        teacher_mutated = self._maybe_mutate_environment(generation, results)
        self._paint_death_pheromones(results)
        self.population = self._reproduce(results)
        return results, teacher_mutated

    def evaluate_vector(self, genome: VectorGenome, generation: int) -> LifeResult:
        route = self._plan_route(genome, generation)
        base_cost = 0.0
        energy = self.config.energy_max
        spent = 0.0
        recharge_visits = 0
        reached_goal = bool(route and route[-1] == GOAL)

        for cell in route[1:]:
            risk = self.risk_at(cell, generation)
            base_cost += 1.0 + risk * genome.risk_multiplier
            shortage = max(0.0, risk - genome.max_risk)
            if shortage:
                spend = self._armor_spend(shortage, genome, energy)
                spent += spend
                energy -= spend
                if spend + 1e-9 < shortage:
                    reached_goal = False
                    base_cost += 50.0 + (shortage - spend) * 12.0
            if self.config.recharge_enabled and cell in self.config.recharge_cells:
                recharge_visits += 1
                energy = min(self.config.energy_max, energy + self.config.recharge_amount)

        optimal_cost = self._optimal_cost_for(genome, generation)
        total_cost = base_cost + spent
        regret = max(0.0, total_cost - optimal_cost)
        fitness = total_cost + self._sharing_penalty(genome)
        if not reached_goal:
            fitness += 100.0

        return LifeResult(
            genome=genome,
            generation=generation,
            route=tuple(route),
            reached_goal=reached_goal,
            base_cost=base_cost,
            energy_spent=spent,
            energy_remaining=energy,
            recharge_visits=recharge_visits,
            regret=regret,
            fitness=fitness,
            niche=genome.niche(),
        )

    def risk_at(self, cell: Position, generation: int) -> float:
        risk = base_risk(cell)
        if cell == (2, 2):
            flicker_phase = (generation // self.config.flicker_period) % 2
            risk = 3.0 if flicker_phase == 0 else 7.0
        risk += self.teacher_spikes.get(cell, 0.0)
        risk += self.pheromone.get(cell, 0.0)
        return risk

    def _armor_spend(self, shortage: float, genome: VectorGenome, energy: float) -> float:
        willingness = genome.armor_bias * (1.0 - 0.65 * genome.conserve_bias)
        if self.config.recharge_enabled and energy < self.config.energy_max * 0.40:
            willingness += genome.recharge_bias * 0.20
        desired = shortage * max(0.0, min(1.0, willingness))
        return min(energy, desired)

    def _plan_route(self, genome: VectorGenome, generation: int) -> list[Position]:
        def neighbors(cell: Position) -> Iterable[Position]:
            row, col = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (row + dr, col + dc)
                if 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE:
                    yield nxt

        def step_cost(cell: Position, energy: float) -> tuple[float, float, int]:
            risk = self.risk_at(cell, generation)
            shortage = max(0.0, risk - genome.max_risk)
            spend = self._armor_spend(shortage, genome, energy)
            remaining = energy - spend
            failure_penalty = 0.0 if spend + 1e-9 >= shortage else 50.0
            if self.config.recharge_enabled and cell in self.config.recharge_cells:
                restored = min(self.config.recharge_amount, self.config.energy_max - remaining)
                remaining += restored
                recharge_reward = -0.45 * restored * genome.recharge_bias
            else:
                recharge_reward = 0.0
            cognitive_load = spend * (1.0 + genome.conserve_bias)
            cost = max(
                0.05,
                1.0
                + risk * genome.risk_multiplier
                + cognitive_load
                + failure_penalty
                + recharge_reward,
            )
            energy_bucket = int(remaining // 2)
            return cost, remaining, energy_bucket

        frontier: list[tuple[float, Position, float, list[Position]]] = [
            (0.0, START, self.config.energy_max, [START])
        ]
        best: dict[tuple[Position, int], float] = {(START, int(self.config.energy_max // 2)): 0.0}

        while frontier:
            cost_so_far, cell, energy, path = heapq.heappop(frontier)
            if cell == GOAL:
                return path
            for nxt in neighbors(cell):
                step, remaining, bucket = step_cost(nxt, energy)
                state = (nxt, bucket)
                new_cost = cost_so_far + step
                if new_cost < best.get(state, math.inf):
                    best[state] = new_cost
                    heapq.heappush(frontier, (new_cost, nxt, remaining, path + [nxt]))
        return []

    def _optimal_cost_for(self, genome: VectorGenome, generation: int) -> float:
        original_bias = genome.armor_bias
        oracle = VectorGenome(
            risk_multiplier=genome.risk_multiplier,
            max_risk=genome.max_risk,
            armor_bias=max(original_bias, 0.95),
            conserve_bias=0.0,
            recharge_bias=1.0,
            ancestry=genome.ancestry,
        )
        route = self._plan_route(oracle, generation)
        if not route:
            return math.inf
        cost = 0.0
        energy = self.config.energy_max
        for cell in route[1:]:
            risk = self.risk_at(cell, generation)
            shortage = max(0.0, risk - oracle.max_risk)
            spend = min(energy, shortage)
            energy -= spend
            cost += 1.0 + risk * oracle.risk_multiplier + spend
            if self.config.recharge_enabled and cell in self.config.recharge_cells:
                energy = min(self.config.energy_max, energy + self.config.recharge_amount)
        return cost

    def _sharing_penalty(self, genome: VectorGenome) -> float:
        neighbors = 0
        for other in self.population:
            if other is genome:
                continue
            distance = math.dist(
                (genome.risk_multiplier, genome.max_risk),
                (other.risk_multiplier, other.max_risk),
            )
            if distance < 0.55:
                neighbors += 1
        return neighbors * self.config.diversity_penalty

    def _maybe_mutate_environment(
        self, generation: int, results: list[LifeResult]
    ) -> bool:
        if generation == 0 or generation % self.config.teacher_interval != 0:
            return False
        recent = self.history[-5:]
        if len(recent) >= 5:
            spread = max(item.average_regret for item in recent) - min(
                item.average_regret for item in recent
            )
            if spread > 0.75:
                return False

        if self.rng.random() < self.config.targeted_teacher_rate:
            dominant = max(
                ("Tank", "Scout", "Hybrid"),
                key=lambda niche: sum(result.niche == niche for result in results),
            )
            candidates = [
                cell
                for result in results
                if result.niche == dominant
                for cell in result.route[1:-1]
            ]
            target = self.rng.choice(candidates) if candidates else (2, 2)
        else:
            target = (self.rng.randrange(GRID_SIZE), self.rng.randrange(GRID_SIZE))
        self.teacher_spikes[target] = min(5.0, self.teacher_spikes.get(target, 0.0) + 2.0)
        return True

    def _paint_death_pheromones(self, results: list[LifeResult]) -> None:
        for result in results:
            if result.reached_goal:
                continue
            for cell in result.route[1:-1]:
                self.pheromone[cell] = min(4.0, self.pheromone.get(cell, 0.0) + 0.15)
        for cell in list(self.pheromone):
            self.pheromone[cell] *= 0.94
            if self.pheromone[cell] < 0.01:
                del self.pheromone[cell]

    def _reproduce(self, results: list[LifeResult]) -> list[VectorGenome]:
        ordered = sorted(results, key=lambda result: result.fitness)
        survivor_count = max(2, int(len(ordered) * self.config.tournament_survival_rate))
        survivors = [result.genome for result in ordered[:survivor_count]]
        next_population = list(survivors)
        while len(next_population) < self.config.population_size:
            parent = self.rng.choice(survivors)
            next_population.append(parent.mutated(self.rng, self.config.mutation_sigma))
        return next_population

    def _summarize(
        self, generation: int, results: list[LifeResult], teacher_mutated: bool
    ) -> GenerationSummary:
        niche_counts = {
            niche: sum(result.niche == niche for result in results)
            for niche in ("Tank", "Scout", "Hybrid")
        }
        return GenerationSummary(
            generation=generation,
            average_regret=fmean(result.regret for result in results),
            best_regret=min(result.regret for result in results),
            average_energy_spent=fmean(result.energy_spent for result in results),
            average_energy_remaining=fmean(result.energy_remaining for result in results),
            diversity=population_diversity(result.genome for result in results),
            teacher_mutated=teacher_mutated,
            flicker_risk=3.0
            if (generation // self.config.flicker_period) % 2 == 0
            else 7.0,
            niche_counts=niche_counts,
        )


def base_risk(cell: Position) -> float:
    """Static risk field before flicker, teacher, and pheromone overlays."""

    row, col = cell
    if col == 0:
        return 2.0
    if col == 4:
        return 6.0
    if (row, col) == (2, 2):
        return 3.0
    if col == 2:
        return 4.0
    return 3.0


def seed_population(size: int, rng: random.Random) -> list[VectorGenome]:
    """Generate a deliberately broad generation zero."""

    population: list[VectorGenome] = []
    for index in range(size):
        fraction = index / max(1, size - 1)
        population.append(
            VectorGenome(
                risk_multiplier=0.1 + fraction * 2.4,
                max_risk=2.0 + (1.0 - fraction) * 7.5,
                armor_bias=rng.random(),
                conserve_bias=rng.random(),
                recharge_bias=rng.random(),
            )
        )
    rng.shuffle(population)
    return population


def population_diversity(genomes: Iterable[VectorGenome]) -> float:
    points = [(genome.risk_multiplier, genome.max_risk) for genome in genomes]
    if len(points) < 2:
        return 0.0
    distances = [
        math.dist(left, right)
        for index, left in enumerate(points)
        for right in points[index + 1 :]
    ]
    return fmean(distances)


def run_experiment(config: SimulationConfig | None = None) -> list[GenerationSummary]:
    """Convenience API used by the CLI, Streamlit app, and tests."""

    experiment = ManifoldExperiment(config or SimulationConfig())
    return experiment.run()
