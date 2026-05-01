"""Core MANIFOLD simulation engine.

The model evolves a population of vectors on top of a 3x3 route manifold.
Value is emergent: vectors discover and re-discover route utility as the
environment changes through teacher interventions and corridor flicker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import random
from typing import Any


ROUTES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

CELL_PARTICIPATION = {
    cell: sum(1 for route in ROUTES if cell in route) for cell in range(9)
}
CELL_BASE_VALUE = {cell: CELL_PARTICIPATION[cell] / 8.0 for cell in range(9)}
CELL_BASE_COST = {cell: 1.0 - CELL_BASE_VALUE[cell] for cell in range(9)}


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class AgentGenome:
    """Evolvable physics for one vector."""

    id: int
    risk_multiplier: float
    max_risk: float
    timing_bias: float
    energy_max: float = 30.0
    energy_per_armor: float = 1.0
    age: int = 0
    last_route: int | None = None
    last_energy_spent: float = 0.0
    last_died: bool = False


@dataclass
class PhaseConfig:
    """Configuration for one experimental phase."""

    name: str
    generations: int
    dual_niche: bool = False
    teacher_enabled: bool = False
    flicker_enabled: bool = False
    ontogeny_enabled: bool = False


@dataclass
class ManifoldConfig:
    """Global simulation hyperparameters."""

    seed: int = 7
    initial_population: int = 30
    min_population: int = 20
    max_population: int = 40
    mutation_sigma: float = 0.05
    fitness_sharing_strength: float = 0.07
    elitism_count: int = 4
    death_penalty: float = 25.0
    pheromone_deposit: float = 1.2
    pheromone_decay: float = 0.90
    spike_decay: float = 0.95
    teacher_interval: int = 15
    teacher_plateau_window: int = 6
    teacher_plateau_eps: float = 0.03
    teacher_targeted_prob: float = 0.70
    teacher_spike_low: float = 1.0
    teacher_spike_high: float = 2.4
    perception_noise_base: float = 0.9
    perception_noise_age_scale: float = 1.5
    energy_max: float = 30.0
    flicker_route: int = 6
    flicker_period: int = 8
    flicker_low: float = 3.0
    flicker_high: float = 7.0
    phases: tuple[PhaseConfig, ...] = (
        PhaseConfig(name="phase_1_static", generations=16),
        PhaseConfig(name="phase_2_dual_niche", generations=18, dual_niche=True),
        PhaseConfig(
            name="phase_3_teacher_flicker",
            generations=24,
            dual_niche=True,
            teacher_enabled=True,
            flicker_enabled=True,
        ),
        PhaseConfig(
            name="phase_4_ontogeny",
            generations=24,
            dual_niche=True,
            teacher_enabled=True,
            flicker_enabled=True,
            ontogeny_enabled=True,
        ),
    )

    @property
    def total_generations(self) -> int:
        return sum(phase.generations for phase in self.phases)


@dataclass
class RouteOutcome:
    """Execution outcome for one route under one agent."""

    route_id: int
    expected_cost: float
    actual_cost: float
    risk_cost: float
    base_cost: float
    energy_spent: float
    died: bool


@dataclass
class GenerationMetrics:
    """Telemetry logged for each generation."""

    generation: int
    phase: str
    population_size: int
    avg_regret: float
    best_regret: float
    diversity: float
    average_energy_spent: float
    death_rate: float
    teacher_event: str | None = None
    niche_counts: dict[str, int] = field(default_factory=dict)
    route_usage: dict[int, int] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Full simulation output."""

    config: ManifoldConfig
    metrics: list[GenerationMetrics]
    final_population: list[AgentGenome]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "metrics": [asdict(metric) for metric in self.metrics],
            "final_population": [asdict(agent) for agent in self.final_population],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class Environment:
    """Mutable non-stationary manifold state."""

    def __init__(self) -> None:
        # Center is safer at baseline; edges/corners are riskier.
        self.base_risk: dict[int, float] = {
            0: 4.2,
            1: 3.6,
            2: 4.2,
            3: 3.6,
            4: 2.8,
            5: 3.6,
            6: 4.2,
            7: 3.6,
            8: 4.2,
        }
        self.pheromone: dict[int, float] = {cell: 0.0 for cell in range(9)}
        self.teacher_spike: dict[int, float] = {cell: 0.0 for cell in range(9)}
        self.route_distance_bias: dict[int, float] = {route_id: 0.0 for route_id in range(8)}
        self.route_risk_bias: dict[int, float] = {route_id: 0.0 for route_id in range(8)}

    def configure_for_phase(self, phase: PhaseConfig) -> None:
        for route_id in range(8):
            self.route_distance_bias[route_id] = 0.0
            self.route_risk_bias[route_id] = 0.0
        if phase.dual_niche:
            # "Scout corridor": longer, safer.
            self.route_distance_bias[1] = 1.8
            self.route_risk_bias[1] = -1.2
            # "Tank corridor": shorter, but dangerous.
            self.route_distance_bias[4] = 0.0
            self.route_risk_bias[4] = 1.6
        else:
            self.route_distance_bias[1] = 0.0
            self.route_risk_bias[1] = 0.0
            self.route_distance_bias[4] = 0.0
            self.route_risk_bias[4] = 0.0

    def route_cells(self, route_id: int) -> tuple[int, int, int]:
        return ROUTES[route_id]

    def flicker_risk(self, generation: int, config: ManifoldConfig) -> float:
        cycle = (generation // config.flicker_period) % 2
        return config.flicker_low if cycle == 0 else config.flicker_high

    def cell_risk(
        self,
        cell: int,
        route_id: int,
        generation: int,
        phase: PhaseConfig,
        config: ManifoldConfig,
    ) -> float:
        risk = self.base_risk[cell] + self.pheromone[cell] + self.teacher_spike[cell]
        risk += self.route_risk_bias[route_id]
        if phase.flicker_enabled and route_id == config.flicker_route:
            risk = self.flicker_risk(generation=generation, config=config)
        return max(0.0, risk)

    def decay(self, config: ManifoldConfig) -> None:
        for cell in range(9):
            self.pheromone[cell] *= config.pheromone_decay
            self.teacher_spike[cell] *= config.spike_decay

    def deposit_death_pheromone(self, route_id: int, config: ManifoldConfig) -> None:
        for cell in self.route_cells(route_id):
            self.pheromone[cell] += config.pheromone_deposit


def _seed_population(config: ManifoldConfig, rng: random.Random) -> list[AgentGenome]:
    population: list[AgentGenome] = []
    for idx in range(config.initial_population):
        ratio = (idx + 0.5) / config.initial_population
        mirrored = 1.0 - ratio
        risk_multiplier = _clip(0.1 + 2.4 * ratio + rng.gauss(0.0, 0.04), 0.1, 2.5)
        max_risk = _clip(2.0 + 7.5 * mirrored + rng.gauss(0.0, 0.16), 2.0, 9.5)
        timing_bias = _clip(rng.random(), 0.0, 1.0)
        population.append(
            AgentGenome(
                id=idx,
                risk_multiplier=risk_multiplier,
                max_risk=max_risk,
                timing_bias=timing_bias,
                energy_max=config.energy_max,
            )
        )
    rng.shuffle(population)
    return population


def _niche_name(agent: AgentGenome) -> str:
    if agent.max_risk <= 3.6 and agent.risk_multiplier >= 1.2:
        return "scout"
    if agent.max_risk >= 6.2 and agent.risk_multiplier <= 1.0:
        return "tank"
    return "hybrid"


def _calculate_diversity(population: list[AgentGenome]) -> float:
    counts: dict[str, int] = {"scout": 0, "tank": 0, "hybrid": 0}
    for agent in population:
        counts[_niche_name(agent)] += 1
    total = len(population)
    entropy = 0.0
    for count in counts.values():
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    # Normalized Shannon entropy (0..1)
    return entropy / math.log(3)


def _perception_noise(agent: AgentGenome, config: ManifoldConfig) -> float:
    return config.perception_noise_base / (1.0 + agent.age / config.perception_noise_age_scale)


def _compute_path_cost(
    agent: AgentGenome,
    route_id: int,
    env: Environment,
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
    *,
    estimate_only: bool,
    rng: random.Random,
) -> RouteOutcome:
    base_cost = sum(CELL_BASE_COST[cell] for cell in env.route_cells(route_id))
    base_cost += env.route_distance_bias[route_id]

    risk_cost = 0.0
    energy_spent = 0.0
    energy_remaining = agent.energy_max if phase.ontogeny_enabled else 0.0
    died = False
    noise = _perception_noise(agent=agent, config=config)

    for cell in env.route_cells(route_id):
        cell_risk = env.cell_risk(
            cell=cell,
            route_id=route_id,
            generation=generation,
            phase=phase,
            config=config,
        )
        used_risk = cell_risk
        if estimate_only:
            used_risk = max(0.0, cell_risk + rng.gauss(0.0, noise))

        required_armor = max(0.0, used_risk - agent.max_risk)
        armor_applied = 0.0
        if phase.ontogeny_enabled:
            # Ontogeny: vectors can spend battery to lower expected risk even
            # before survival pressure becomes critical.
            proactive_fraction = _clip(
                0.10 + 0.50 * agent.timing_bias + 0.15 * agent.risk_multiplier,
                0.0,
                0.85,
            )
            target_residual = max(0.0, used_risk * (1.0 - proactive_fraction))
            desired_armor = max(0.0, used_risk - target_residual)
            armor_budget_need = max(required_armor, desired_armor)

            affordable_armor = energy_remaining / agent.energy_per_armor
            armor_applied = min(armor_budget_need, affordable_armor)
            spent = armor_applied * agent.energy_per_armor
            energy_spent += spent
            energy_remaining -= spent

        residual_risk = used_risk - armor_applied
        risk_cost += residual_risk * agent.risk_multiplier
        if residual_risk > agent.max_risk:
            died = True
            break

    timing_adjustment = 0.0
    if phase.flicker_enabled and route_id == config.flicker_route:
        flicker_is_high = env.flicker_risk(generation=generation, config=config) > (
            config.flicker_low + config.flicker_high
        ) / 2
        if flicker_is_high:
            timing_adjustment = 2.2 * agent.timing_bias
        else:
            timing_adjustment = -1.2 * agent.timing_bias

    total_cost = base_cost + risk_cost + timing_adjustment
    if phase.ontogeny_enabled:
        total_cost += energy_spent
    if died:
        total_cost += config.death_penalty

    return RouteOutcome(
        route_id=route_id,
        expected_cost=total_cost if estimate_only else 0.0,
        actual_cost=total_cost if not estimate_only else 0.0,
        risk_cost=risk_cost,
        base_cost=base_cost,
        energy_spent=energy_spent,
        died=died,
    )


def _evaluate_agent(
    agent: AgentGenome,
    env: Environment,
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
    rng: random.Random,
) -> tuple[float, RouteOutcome, float]:
    estimated_outcomes: list[RouteOutcome] = []
    for route_id in range(len(ROUTES)):
        estimated_outcomes.append(
            _compute_path_cost(
                agent=agent,
                route_id=route_id,
                env=env,
                generation=generation,
                phase=phase,
                config=config,
                estimate_only=True,
                rng=rng,
            )
        )

    chosen = min(estimated_outcomes, key=lambda outcome: outcome.expected_cost)
    actual_chosen = _compute_path_cost(
        agent=agent,
        route_id=chosen.route_id,
        env=env,
        generation=generation,
        phase=phase,
        config=config,
        estimate_only=False,
        rng=rng,
    )

    optimal_actual_cost = min(
        _compute_path_cost(
            agent=agent,
            route_id=route_id,
            env=env,
            generation=generation,
            phase=phase,
            config=config,
            estimate_only=False,
            rng=rng,
        ).actual_cost
        for route_id in range(len(ROUTES))
    )
    regret = actual_chosen.actual_cost - optimal_actual_cost
    return regret, actual_chosen, optimal_actual_cost


def _mutate_agent(
    parent: AgentGenome,
    child_id: int,
    config: ManifoldConfig,
    rng: random.Random,
) -> AgentGenome:
    sigma = config.mutation_sigma
    return AgentGenome(
        id=child_id,
        risk_multiplier=_clip(parent.risk_multiplier + rng.gauss(0.0, sigma), 0.1, 2.5),
        max_risk=_clip(parent.max_risk + rng.gauss(0.0, sigma * 3.0), 2.0, 9.5),
        timing_bias=_clip(parent.timing_bias + rng.gauss(0.0, sigma * 1.4), 0.0, 1.0),
        energy_max=parent.energy_max,
        energy_per_armor=parent.energy_per_armor,
    )


def _should_trigger_teacher(
    metrics: list[GenerationMetrics],
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
) -> bool:
    if not phase.teacher_enabled:
        return False
    if generation == 0 or generation % config.teacher_interval != 0:
        return False
    if len(metrics) < config.teacher_plateau_window:
        return False
    window = metrics[-config.teacher_plateau_window :]
    regret_values = [m.avg_regret for m in window]
    return max(regret_values) - min(regret_values) <= config.teacher_plateau_eps


def _apply_teacher(
    env: Environment,
    population: list[AgentGenome],
    route_usage: dict[int, int],
    config: ManifoldConfig,
    rng: random.Random,
) -> str:
    if not population:
        return "teacher:no_population"

    dominant_niche_counts: dict[str, int] = {"scout": 0, "tank": 0, "hybrid": 0}
    for agent in population:
        dominant_niche_counts[_niche_name(agent)] += 1
    dominant_niche = max(dominant_niche_counts, key=dominant_niche_counts.get)

    if rng.random() <= config.teacher_targeted_prob:
        # Targeted attack hits the currently dominant route.
        dominant_route = max(route_usage, key=route_usage.get)
        spike = rng.uniform(config.teacher_spike_low, config.teacher_spike_high)
        for cell in env.route_cells(dominant_route):
            env.teacher_spike[cell] += spike
        return f"targeted:{dominant_niche}:route_{dominant_route}"

    random_cell = rng.randrange(0, 9)
    spike = rng.uniform(config.teacher_spike_low, config.teacher_spike_high)
    env.teacher_spike[random_cell] += spike
    return f"random:cell_{random_cell}"


def _select_next_population(
    population: list[AgentGenome],
    adjusted_regrets: dict[int, float],
    config: ManifoldConfig,
    rng: random.Random,
    id_counter_start: int,
) -> tuple[list[AgentGenome], int]:
    ordered = sorted(population, key=lambda agent: adjusted_regrets[agent.id])
    elite_count = min(config.elitism_count, len(ordered))
    elites = ordered[:elite_count]

    next_population: list[AgentGenome] = []
    for elite in elites:
        next_population.append(
            AgentGenome(
                id=elite.id,
                risk_multiplier=elite.risk_multiplier,
                max_risk=elite.max_risk,
                timing_bias=elite.timing_bias,
                energy_max=elite.energy_max,
                energy_per_armor=elite.energy_per_armor,
                age=elite.age + 1,
            )
        )

    target_size = rng.randint(config.min_population, config.max_population)
    parent_weights: list[float] = []
    for agent in ordered:
        score = 1.0 / (1.0 + adjusted_regrets[agent.id])
        parent_weights.append(max(score, 1e-6))

    next_id = id_counter_start
    while len(next_population) < target_size:
        parent = rng.choices(ordered, weights=parent_weights, k=1)[0]
        child = _mutate_agent(parent=parent, child_id=next_id, config=config, rng=rng)
        next_population.append(child)
        next_id += 1

    return next_population, next_id


def run_manifold(config: ManifoldConfig | None = None) -> SimulationResult:
    """Run full MANIFOLD simulation and return all telemetry."""

    if config is None:
        config = ManifoldConfig()
    rng = random.Random(config.seed)
    env = Environment()
    population = _seed_population(config=config, rng=rng)

    metrics: list[GenerationMetrics] = []
    id_counter = max(agent.id for agent in population) + 1
    absolute_generation = 0

    for phase in config.phases:
        env.configure_for_phase(phase)
        for _ in range(phase.generations):
            regrets: dict[int, float] = {}
            route_usage: dict[int, int] = {route_id: 0 for route_id in range(len(ROUTES))}
            deaths = 0
            total_energy = 0.0

            for agent in population:
                regret, outcome, _ = _evaluate_agent(
                    agent=agent,
                    env=env,
                    generation=absolute_generation,
                    phase=phase,
                    config=config,
                    rng=rng,
                )
                regrets[agent.id] = regret
                route_usage[outcome.route_id] += 1
                agent.last_route = outcome.route_id
                agent.last_energy_spent = outcome.energy_spent
                agent.last_died = outcome.died
                total_energy += outcome.energy_spent
                if outcome.died:
                    deaths += 1
                    env.deposit_death_pheromone(route_id=outcome.route_id, config=config)

            niche_counts: dict[str, int] = {"scout": 0, "tank": 0, "hybrid": 0}
            for agent in population:
                niche_counts[_niche_name(agent)] += 1

            adjusted_regrets: dict[int, float] = {}
            for agent in population:
                share = 1.0 + config.fitness_sharing_strength * (
                    niche_counts[_niche_name(agent)] - 1
                )
                adjusted_regrets[agent.id] = max(0.0, regrets[agent.id]) * share

            avg_regret = sum(regrets.values()) / len(population)
            best_regret = min(regrets.values())
            diversity = _calculate_diversity(population=population)
            avg_energy = total_energy / len(population)
            death_rate = deaths / len(population)

            teacher_event: str | None = None
            if _should_trigger_teacher(
                metrics=metrics,
                generation=absolute_generation,
                phase=phase,
                config=config,
            ):
                teacher_event = _apply_teacher(
                    env=env,
                    population=population,
                    route_usage=route_usage,
                    config=config,
                    rng=rng,
                )

            metrics.append(
                GenerationMetrics(
                    generation=absolute_generation,
                    phase=phase.name,
                    population_size=len(population),
                    avg_regret=avg_regret,
                    best_regret=best_regret,
                    diversity=diversity,
                    average_energy_spent=avg_energy,
                    death_rate=death_rate,
                    teacher_event=teacher_event,
                    niche_counts=niche_counts,
                    route_usage=route_usage,
                )
            )

            population, id_counter = _select_next_population(
                population=population,
                adjusted_regrets=adjusted_regrets,
                config=config,
                rng=rng,
                id_counter_start=id_counter,
            )
            env.decay(config=config)
            absolute_generation += 1

    return SimulationResult(config=config, metrics=metrics, final_population=population)


def summarize_result(result: SimulationResult) -> dict[str, Any]:
    """Compact summary for CLI/UI usage."""

    final = result.metrics[-1]
    by_phase: dict[str, dict[str, float]] = {}
    for phase in result.config.phases:
        phase_metrics = [m for m in result.metrics if m.phase == phase.name]
        by_phase[phase.name] = {
            "avg_regret_start": phase_metrics[0].avg_regret,
            "avg_regret_end": phase_metrics[-1].avg_regret,
            "diversity_end": phase_metrics[-1].diversity,
            "energy_end": phase_metrics[-1].average_energy_spent,
            "death_rate_end": phase_metrics[-1].death_rate,
        }
    return {
        "total_generations": result.config.total_generations,
        "final_population": len(result.final_population),
        "final_avg_regret": final.avg_regret,
        "final_best_regret": final.best_regret,
        "final_diversity": final.diversity,
        "final_energy_spent": final.average_energy_spent,
        "phase_summary": by_phase,
    }
