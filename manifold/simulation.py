"""Core MANIFOLD simulation engine.

MANIFOLD separates two learning timescales on a non-stationary grid:
phylogeny evolves body parameters such as risk tolerance, while ontogeny evolves
within-life policy parameters such as energy aversion, charger use, signalling,
and verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from statistics import fmean
from typing import Iterable, Literal


Position = tuple[int, int]
TeacherMode = Literal["periodic", "reactive", "random", "adversarial", "multi"]

DEFAULT_GRID_SIZE = 11
DEFAULT_ENERGY_MAX = 8.0
DEFAULT_CHARGERS: tuple[Position, ...] = ((3, 5), (7, 5))


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a MANIFOLD experiment."""

    population_size: int = 52
    generations: int = 200
    seed: int = 13
    grid_size: int = DEFAULT_GRID_SIZE
    energy_max: float = DEFAULT_ENERGY_MAX
    teacher_mode: TeacherMode = "periodic"
    teacher_interval: int = 20
    teacher_duration: int = 5
    teacher_strength: float = 4.0
    random_teacher_rate: float = 0.30
    recharge_enabled: bool = True
    recharge_amount: float = 4.0
    recharge_cells: tuple[Position, ...] = DEFAULT_CHARGERS
    communication_enabled: bool = False
    signal_cost: float = 0.5
    tournament_survival_rate: float = 0.55
    diversity_penalty: float = 0.04
    rm_sigma: float = 0.02
    max_r_sigma: float = 0.02
    aversion_sigma: float = 0.12
    policy_sigma: float = 0.12

    @property
    def start(self) -> Position:
        return (0, self.grid_size // 2)

    @property
    def goal(self) -> Position:
        return (self.grid_size - 1, self.grid_size // 2)


@dataclass
class VectorGenome:
    """Heritable body and policy traits for one vector agent."""

    risk_multiplier: float
    max_risk: float
    energy_aversion: float
    charger_bias: float = 0.5
    signal_honesty: float = 0.9
    verification_bias: float = 0.2
    ancestry: str = "seed"

    def niche(self) -> str:
        if self.max_risk >= 5.9:
            return "Body"
        if self.energy_aversion >= 1.8 and self.charger_bias >= 0.55:
            return "Planner"
        return "Hybrid"

    def mutated(self, rng: random.Random, config: SimulationConfig) -> "VectorGenome":
        def clamp(value: float, low: float, high: float) -> float:
            return max(low, min(high, value))

        return VectorGenome(
            risk_multiplier=clamp(
                self.risk_multiplier + rng.gauss(0.0, config.rm_sigma), 0.1, 1.0
            ),
            max_risk=clamp(self.max_risk + rng.gauss(0.0, config.max_r_sigma), 4.0, 6.5),
            energy_aversion=clamp(
                self.energy_aversion + rng.gauss(0.0, config.aversion_sigma), 0.5, 2.5
            ),
            charger_bias=clamp(
                self.charger_bias + rng.gauss(0.0, config.policy_sigma), 0.0, 1.0
            ),
            signal_honesty=clamp(
                self.signal_honesty + rng.gauss(0.0, config.policy_sigma), 0.0, 1.0
            ),
            verification_bias=clamp(
                self.verification_bias + rng.gauss(0.0, config.policy_sigma), 0.0, 1.0
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
    terrain_cost: float
    teacher_cost: float
    waste_cost: float
    total_cost: float
    energy_spent: float
    energy_remaining: float
    recharge_visits: int
    signal: str
    trusted_signal: bool
    regret: float
    fitness: float
    niche: str


@dataclass(frozen=True)
class GenerationSummary:
    """Population-level metrics emitted after each generation."""

    generation: int
    average_regret: float
    best_regret: float
    survival_rate: float
    average_energy_spent: float
    average_energy_remaining: float
    average_recharge_visits: float
    average_max_risk: float
    average_energy_aversion: float
    signal_spike_correlation: float
    lie_rate: float
    diversity: float
    teacher_mutated: bool
    spike_active: bool
    teacher_strength: float
    niche_counts: dict[str, int]

    @property
    def average_waste_cost(self) -> float:
        return self.average_energy_spent * self.average_energy_aversion


@dataclass
class ManifoldExperiment:
    """Runnable MANIFOLD experiment state."""

    config: SimulationConfig = field(default_factory=SimulationConfig)
    rng: random.Random = field(init=False)
    population: list[VectorGenome] = field(init=False)
    pheromone: dict[Position, float] = field(default_factory=dict)
    teacher_spikes: dict[Position, float] = field(default_factory=dict)
    history: list[GenerationSummary] = field(default_factory=list)
    teacher_strengths: dict[str, float] = field(
        default_factory=lambda: {"periodic": 2.12, "reactive": 2.42, "random": 2.59}
    )

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.population = seed_population(self.config.population_size, self.rng)

    def run(self) -> list[GenerationSummary]:
        for generation in range(self.config.generations):
            results, teacher_mutated = self.step(generation)
            self.history.append(self._summarize(generation, results, teacher_mutated))
        return self.history

    def step(self, generation: int) -> tuple[list[LifeResult], bool]:
        teacher_mutated = self._update_teacher(generation)
        results = [self.evaluate_vector(genome, generation) for genome in self.population]
        self._paint_death_pheromones(results)
        self._adapt_teachers(results)
        self.population = self._reproduce(results)
        return results, teacher_mutated

    def evaluate_vector(self, genome: VectorGenome, generation: int) -> LifeResult:
        signal = self._emit_signal(genome, generation)
        trusted_signal = self._trust_signal(genome, signal)
        route = self._plan_route(genome, generation, trusted_signal)
        energy = self.config.energy_max
        terrain_cost = 0.0
        teacher_cost = 0.0
        waste_cost = self.config.signal_cost if self.config.communication_enabled else 0.0
        spent = waste_cost
        recharge_visits = 0
        reached_goal = bool(route and route[-1] == self.config.goal)

        if spent > energy:
            reached_goal = False
        energy = max(0.0, energy - spent)

        for cell in route[1:]:
            terrain, teacher = self._risk_components(cell, generation)
            risk = terrain + teacher + self.pheromone.get(cell, 0.0)
            terrain_cost += 1.0 + terrain * genome.risk_multiplier
            teacher_cost += teacher * genome.risk_multiplier
            needed = boost_needed(risk, genome.max_risk)
            if needed:
                spend = min(energy, needed)
                spent += spend
                energy -= spend
                waste_cost += spend * genome.energy_aversion
                if spend + 1e-9 < needed:
                    reached_goal = False
                    teacher_cost += 25.0 + (needed - spend) * 6.0
            if self.config.recharge_enabled and cell in self.config.recharge_cells:
                recharge_visits += 1
                energy = min(self.config.energy_max, energy + self.config.recharge_amount)

        total_cost = terrain_cost + teacher_cost + waste_cost
        optimal_cost = self._oracle_cost(genome, generation)
        regret = max(0.0, total_cost - optimal_cost)
        fitness = total_cost + self._sharing_penalty(genome)
        if not reached_goal:
            fitness += 100.0

        return LifeResult(
            genome=genome,
            generation=generation,
            route=tuple(route),
            reached_goal=reached_goal,
            terrain_cost=terrain_cost,
            teacher_cost=teacher_cost,
            waste_cost=waste_cost,
            total_cost=total_cost,
            energy_spent=spent,
            energy_remaining=energy,
            recharge_visits=recharge_visits,
            signal=signal,
            trusted_signal=trusted_signal,
            regret=regret,
            fitness=fitness,
            niche=genome.niche(),
        )

    def risk_at(self, cell: Position, generation: int) -> float:
        terrain, teacher = self._risk_components(cell, generation)
        return terrain + teacher + self.pheromone.get(cell, 0.0)

    def _risk_components(self, cell: Position, generation: int) -> tuple[float, float]:
        terrain = base_risk(cell, self.config.grid_size)
        teacher = self.teacher_spikes.get(cell, 0.0)
        return terrain, teacher

    def _spike_active(self, generation: int) -> bool:
        mode = self.config.teacher_mode
        if mode == "periodic":
            return generation % self.config.teacher_interval < self.config.teacher_duration
        if mode == "reactive":
            return bool(self.history and self.history[-1].survival_rate > 0.80)
        if mode == "random":
            return self.rng.random() < self.config.random_teacher_rate
        if mode == "adversarial":
            return bool(self.history and self.history[-1].average_recharge_visits > 1.2)
        if mode == "multi":
            periodic = generation % self.config.teacher_interval < self.config.teacher_duration
            reactive = bool(self.history and self.history[-1].survival_rate > 0.80)
            random_spike = self.rng.random() < self.config.random_teacher_rate
            return periodic or reactive or random_spike
        return False

    def _emit_signal(self, genome: VectorGenome, generation: int) -> str:
        if not self.config.communication_enabled:
            return "00"
        spike_soon = any(self._spike_active(generation + offset) for offset in range(1, 5))
        honest_signal = "10" if spike_soon else "00"
        if self.rng.random() <= genome.signal_honesty:
            return honest_signal
        return "10" if honest_signal == "00" else "00"

    def _trust_signal(self, genome: VectorGenome, signal: str) -> bool:
        if not self.config.communication_enabled or signal != "10":
            return False
        return self.rng.random() > genome.verification_bias * 0.35

    def _plan_route(
        self, genome: VectorGenome, generation: int, trusted_signal: bool = False
    ) -> list[Position]:
        def step_cost(cell: Position, energy: float) -> tuple[float, float, int]:
            risk = self.risk_at(cell, generation)
            if trusted_signal and cell in bottleneck_cells(self.config.grid_size):
                risk += 1.0
            needed = boost_needed(risk, genome.max_risk)
            spend = min(energy, needed)
            remaining = energy - spend
            failure_penalty = 0.0 if spend + 1e-9 >= needed else 50.0
            recharge_reward = 0.0
            if self.config.recharge_enabled and cell in self.config.recharge_cells:
                restored = min(self.config.recharge_amount, self.config.energy_max - remaining)
                remaining += restored
                if needed > 0 or trusted_signal or bool(self.teacher_spikes):
                    planned_value = max(restored, self.config.recharge_amount * 0.75)
                    recharge_reward = -planned_value * genome.charger_bias
                else:
                    recharge_reward = restored * (1.0 - genome.charger_bias) * 0.25
            cost = max(
                0.05,
                1.0
                + risk * genome.risk_multiplier
                + spend * genome.energy_aversion
                + failure_penalty
                + recharge_reward,
            )
            return cost, remaining, int(remaining * 2)

        frontier: list[tuple[float, Position, float, list[Position]]] = [
            (0.0, self.config.start, self.config.energy_max, [self.config.start])
        ]
        best: dict[tuple[Position, int], float] = {
            (self.config.start, int(self.config.energy_max * 2)): 0.0
        }
        while frontier:
            cost_so_far, cell, energy, path = heapq.heappop(frontier)
            if cell == self.config.goal:
                return path
            for nxt in neighbors(cell, self.config.grid_size):
                step, remaining, bucket = step_cost(nxt, energy)
                state = (nxt, bucket)
                new_cost = cost_so_far + step
                if new_cost < best.get(state, math.inf):
                    best[state] = new_cost
                    heapq.heappush(frontier, (new_cost, nxt, remaining, path + [nxt]))
        return []

    def _oracle_cost(self, genome: VectorGenome, generation: int) -> float:
        oracle = VectorGenome(
            risk_multiplier=genome.risk_multiplier,
            max_risk=genome.max_risk,
            energy_aversion=0.5,
            charger_bias=1.0,
            signal_honesty=1.0,
            verification_bias=1.0,
            ancestry=genome.ancestry,
        )
        route = self._plan_route(oracle, generation, trusted_signal=True)
        if not route:
            return math.inf
        energy = self.config.energy_max
        cost = 0.0
        for cell in route[1:]:
            terrain, teacher = self._risk_components(cell, generation)
            risk = terrain + teacher
            needed = boost_needed(risk, oracle.max_risk)
            spend = min(energy, needed)
            energy -= spend
            cost += 1.0 + terrain * oracle.risk_multiplier + teacher * oracle.risk_multiplier + spend
            if self.config.recharge_enabled and cell in self.config.recharge_cells:
                energy = min(self.config.energy_max, energy + self.config.recharge_amount)
        return cost

    def _sharing_penalty(self, genome: VectorGenome) -> float:
        neighbors_count = 0
        for other in self.population:
            if other is genome:
                continue
            distance = math.dist(
                (genome.risk_multiplier, genome.max_risk, genome.energy_aversion),
                (other.risk_multiplier, other.max_risk, other.energy_aversion),
            )
            if distance < 0.35:
                neighbors_count += 1
        return neighbors_count * self.config.diversity_penalty

    def _update_teacher(self, generation: int) -> bool:
        self.teacher_spikes.clear()
        mutated = self._spike_active(generation)
        if not mutated:
            return False
        for cell in bottleneck_cells(self.config.grid_size):
            self.teacher_spikes[cell] = self.config.teacher_strength
        return True

    def _adapt_teachers(self, results: list[LifeResult]) -> None:
        if self.config.teacher_mode != "multi" or not results:
            return
        survival = sum(result.reached_goal for result in results) / len(results)
        recharge = fmean(result.recharge_visits for result in results)
        self.teacher_strengths["periodic"] += 0.01 if survival > 0.85 else -0.005
        self.teacher_strengths["reactive"] += 0.01 if survival > 0.80 else -0.005
        self.teacher_strengths["random"] += 0.01 if recharge > 1.0 else -0.003
        for name, value in list(self.teacher_strengths.items()):
            self.teacher_strengths[name] = max(1.0, min(4.0, value))

    def _paint_death_pheromones(self, results: list[LifeResult]) -> None:
        for result in results:
            if result.reached_goal:
                continue
            for cell in result.route[1:-1]:
                self.pheromone[cell] = min(2.0, self.pheromone.get(cell, 0.0) + 0.08)
        for cell in list(self.pheromone):
            self.pheromone[cell] *= 0.92
            if self.pheromone[cell] < 0.01:
                del self.pheromone[cell]

    def _reproduce(self, results: list[LifeResult]) -> list[VectorGenome]:
        ordered = sorted(results, key=lambda result: result.fitness)
        survivor_count = max(2, int(len(ordered) * self.config.tournament_survival_rate))
        survivors = [result.genome for result in ordered[:survivor_count]]
        next_population = list(survivors)
        while len(next_population) < self.config.population_size:
            parent = self.rng.choice(survivors)
            next_population.append(parent.mutated(self.rng, self.config))
        return next_population

    def _summarize(
        self, generation: int, results: list[LifeResult], teacher_mutated: bool
    ) -> GenerationSummary:
        niche_counts = {
            niche: sum(result.niche == niche for result in results)
            for niche in ("Body", "Planner", "Hybrid")
        }
        spike_labels = [1 if self._spike_active(result.generation) else 0 for result in results]
        signal_labels = [1 if result.signal == "10" else 0 for result in results]
        return GenerationSummary(
            generation=generation,
            average_regret=fmean(result.regret for result in results),
            best_regret=min(result.regret for result in results),
            survival_rate=sum(result.reached_goal for result in results) / len(results),
            average_energy_spent=fmean(result.energy_spent for result in results),
            average_energy_remaining=fmean(result.energy_remaining for result in results),
            average_recharge_visits=fmean(result.recharge_visits for result in results),
            average_max_risk=fmean(result.genome.max_risk for result in results),
            average_energy_aversion=fmean(result.genome.energy_aversion for result in results),
            signal_spike_correlation=binary_correlation(signal_labels, spike_labels),
            lie_rate=fmean(
                1.0
                if self.config.communication_enabled
                and ((result.signal == "10") != bool(spike_labels[index]))
                else 0.0
                for index, result in enumerate(results)
            ),
            diversity=population_diversity(result.genome for result in results),
            teacher_mutated=teacher_mutated,
            spike_active=teacher_mutated,
            teacher_strength=max(self.teacher_strengths.values())
            if self.config.teacher_mode == "multi"
            else self.config.teacher_strength,
            niche_counts=niche_counts,
        )


def boost_needed(risk: float, max_risk: float) -> float:
    """Energy needed to cross a cell above body tolerance."""

    return max(0.0, risk - max_risk) * 2.0


def base_risk(cell: Position, grid_size: int = DEFAULT_GRID_SIZE) -> float:
    """Static risk field: one calm corridor, expensive detours, teacher bottleneck."""

    row, col = cell
    mid = grid_size // 2
    if col == mid:
        return 5.0
    if abs(col - mid) == 1:
        return 6.5
    if col in (0, grid_size - 1):
        return 8.0
    return 7.0


def bottleneck_cells(grid_size: int = DEFAULT_GRID_SIZE) -> tuple[Position, ...]:
    mid = grid_size // 2
    center = grid_size // 2
    return ((center, mid),)


def neighbors(cell: Position, grid_size: int) -> Iterable[Position]:
    row, col = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (row + dr, col + dc)
        if 0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size:
            yield nxt


def seed_population(size: int, rng: random.Random) -> list[VectorGenome]:
    """Generate a broad generation zero across MANIFOLD trait bounds."""

    population: list[VectorGenome] = []
    for index in range(size):
        fraction = index / max(1, size - 1)
        population.append(
            VectorGenome(
                risk_multiplier=0.1 + fraction * 0.9,
                max_risk=4.0 + rng.random() * 2.5,
                energy_aversion=0.5 + (1.0 - fraction) * 2.0,
                charger_bias=rng.random(),
                signal_honesty=0.85 + rng.random() * 0.15,
                verification_bias=rng.random() * 0.4,
            )
        )
    rng.shuffle(population)
    return population


def population_diversity(genomes: Iterable[VectorGenome]) -> float:
    points = [
        (genome.risk_multiplier, genome.max_risk, genome.energy_aversion)
        for genome in genomes
    ]
    if len(points) < 2:
        return 0.0
    distances = [
        math.dist(left, right)
        for index, left in enumerate(points)
        for right in points[index + 1 :]
    ]
    return fmean(distances)


def binary_correlation(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var == 0.0 or right_var == 0.0:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)


def transfer_population(
    source: ManifoldExperiment, target_grid_size: int = 21
) -> ManifoldExperiment:
    """Create a larger-world experiment seeded by an evolved population."""

    target = ManifoldExperiment(
        SimulationConfig(
            population_size=len(source.population),
            generations=source.config.generations,
            seed=source.config.seed + 1,
            grid_size=target_grid_size,
            energy_max=source.config.energy_max,
            teacher_mode=source.config.teacher_mode,
            recharge_enabled=source.config.recharge_enabled,
            recharge_amount=source.config.recharge_amount,
            recharge_cells=scaled_chargers(target_grid_size),
            communication_enabled=source.config.communication_enabled,
        )
    )
    target.population = [genome.mutated(target.rng, target.config) for genome in source.population]
    return target


def scaled_chargers(grid_size: int) -> tuple[Position, ...]:
    mid = grid_size // 2
    return ((max(1, grid_size // 3), mid), (min(grid_size - 2, 2 * grid_size // 3), mid))


def run_experiment(config: SimulationConfig | None = None) -> list[GenerationSummary]:
    """Convenience API used by the CLI, Streamlit app, and tests."""

    experiment = ManifoldExperiment(config or SimulationConfig())
    return experiment.run()
