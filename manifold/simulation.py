"""Core MANIFOLD simulation engine.

This version supports:
- classic evolutionary path adaptation (v1-v5),
- explicit 3-layer grid dynamics (physical/info/social),
- adaptive rule penalties,
- live event ingestion,
- predator self-tuning,
- explainability traces and memory-market economics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from .connectors import ConnectorEvent, load_connector_events
from .rules import RuleDefinition, compile_rulebook


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
RECHARGE_CELLS: dict[int, float] = {
    1: 5.5,
    4: 8.0,
    7: 5.5,
}
ACTION_ADVANCE = "advance"
ACTION_PAUSE_RECHARGE = "pause_recharge"
ACTION_DETOUR_RECHARGE = "detour_recharge"

LAYER_PHYSICAL = "physical"
LAYER_INFO = "information"
LAYER_SOCIAL = "social"
LAYER_NAMES: tuple[str, ...] = (LAYER_PHYSICAL, LAYER_INFO, LAYER_SOCIAL)

DEFAULT_RULEBOOK = """
if late_delivery then -£8.20 @target=0.18 @alpha=1.25 @min=0.8 @max=35
if skip_verification then -£5.40 @target=0.22 @alpha=1.05 @min=0.5 @max=32
if low_inventory then -£6.80 @target=0.20 @alpha=1.10 @min=0.7 @max=38
"""

CELL_PARTICIPATION = {
    cell: sum(1 for route in ROUTES if cell in route) for cell in range(9)
}
CELL_BASE_VALUE = {cell: CELL_PARTICIPATION[cell] / 8.0 for cell in range(9)}
CELL_BASE_COST = {cell: 1.0 - CELL_BASE_VALUE[cell] for cell in range(9)}


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class AgentGenome:
    """Evolvable physics and behavioral biases for one vector."""

    id: int
    risk_multiplier: float
    max_risk: float
    timing_bias: float
    recharge_bias: float
    honesty_bias: float
    verification_skill: float
    reputation: float = 0.50
    verified_memory: float = 0.0
    energy_max: float = 30.0
    energy_per_armor: float = 1.0
    age: int = 0
    last_route: int | None = None
    last_action: str = ACTION_ADVANCE
    last_energy_spent: float = 0.0
    last_recharge_gained: float = 0.0
    last_memory_revenue: float = 0.0
    last_died: bool = False
    last_explanation: str = ""


@dataclass
class PhaseConfig:
    """Configuration for one experimental phase."""

    name: str
    generations: int
    dual_niche: bool = False
    teacher_enabled: bool = False
    flicker_enabled: bool = False
    ontogeny_enabled: bool = False
    recharge_enabled: bool = False
    energy_budget: float | None = None
    multi_layer_enabled: bool = False
    adaptive_rules_enabled: bool = False
    memory_market_enabled: bool = False
    predator_auto_tuning: bool = False
    rule_targets_enabled: bool = False


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
    recharge_reward_scale: float = 1.0
    pause_penalty: float = 0.85
    detour_penalty: float = 1.35
    recharge_flicker_period: int = 6
    recharge_flicker_low: float = 0.65
    recharge_flicker_high: float = 1.25
    flicker_route: int = 6
    flicker_period: int = 8
    flicker_low: float = 3.0
    flicker_high: float = 7.0
    layer_coupling_info_to_physical: float = 0.45
    layer_coupling_social_to_info: float = 0.35
    rulebook_text: str = DEFAULT_RULEBOOK
    predator_spawn_rate: float = 0.05
    connector_events_path: str | None = None
    transfer_neutrality_path: str | None = None
    use_generation_cache: bool = True
    use_vectorized_scoring: bool = True
    use_event_indexing: bool = True
    target_intent: str = "Find the cheapest trustworthy supplier under £50"
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
            energy_budget=30.0,
        ),
        PhaseConfig(
            name="phase_5_recharge_hierarchical",
            generations=24,
            dual_niche=True,
            teacher_enabled=True,
            flicker_enabled=True,
            ontogeny_enabled=True,
            recharge_enabled=True,
            energy_budget=12.0,
        ),
        PhaseConfig(
            name="phase_6_production_stack",
            generations=28,
            dual_niche=True,
            teacher_enabled=True,
            flicker_enabled=True,
            ontogeny_enabled=True,
            recharge_enabled=True,
            energy_budget=12.0,
            multi_layer_enabled=True,
            adaptive_rules_enabled=True,
            memory_market_enabled=True,
            predator_auto_tuning=True,
            rule_targets_enabled=True,
        ),
    )

    @property
    def total_generations(self) -> int:
        return sum(phase.generations for phase in self.phases)


@dataclass
class RouteOutcome:
    """Execution outcome for one route under one action plan."""

    route_id: int
    action: str
    expected_cost: float
    actual_cost: float
    risk_cost: float
    base_cost: float
    energy_spent: float
    recharge_gained: float
    memory_revenue: float
    rule_penalty_cost: float
    layer_costs: dict[str, float]
    break_flags: dict[str, int]
    died: bool
    explain_log: str


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
    average_recharge_gained: float
    average_memory_revenue: float
    recharge_event_rate: float
    death_rate: float
    predator_spawn_rate: float
    max_reputation: float
    layer_regret_contrib: dict[str, float] = field(default_factory=dict)
    rule_break_rates: dict[str, float] = field(default_factory=dict)
    rule_penalties: dict[str, float] = field(default_factory=dict)
    teacher_event: str | None = None
    niche_counts: dict[str, int] = field(default_factory=dict)
    route_usage: dict[int, int] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    confidence_distribution: dict[int, float] = field(default_factory=dict)
    target_intent: str = ""
    explain_samples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RouteCellCache:
    """Cached per-cell values for one route in one generation."""

    combined_risk: float
    info_noise: float
    layer_physical: float
    layer_info: float
    layer_social: float


@dataclass
class GenerationEvaluationCache:
    """Cached route-level values reused across many agents."""

    route_base_cost: dict[int, float]
    route_cells: dict[int, tuple[RouteCellCache, ...]]
    route_actions: dict[int, tuple[str, ...]]
    route_target_penalty: dict[int, float]


@dataclass(frozen=True)
class GenerationContext:
    """Per-generation static caches for faster agent evaluation."""

    route_cells: tuple[tuple[int, int, int], ...]
    route_base_cost: tuple[float, ...]
    candidate_pairs: tuple[tuple[int, str], ...]


@dataclass
class SimulationResult:
    """Full simulation output."""

    config: ManifoldConfig
    metrics: list[GenerationMetrics]
    final_population: list[AgentGenome]
    transfer_artifact: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "metrics": [asdict(metric) for metric in self.metrics],
            "final_population": [asdict(agent) for agent in self.final_population],
            "transfer_artifact": self.transfer_artifact,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _index_events_by_generation(
    events: tuple[ConnectorEvent, ...],
) -> dict[int, tuple[ConnectorEvent, ...]]:
    indexed: dict[int, list[ConnectorEvent]] = {}
    for event in events:
        indexed.setdefault(event.generation, []).append(event)
    return {
        generation: tuple(bucket) for generation, bucket in indexed.items()
    }


def _prepare_generation_context(
    *,
    env: LayeredState,
    phase: PhaseConfig,
) -> GenerationContext:
    route_cells = tuple(env.route_cells(route_id) for route_id in range(len(ROUTES)))
    route_base_cost = tuple(
        sum(CELL_BASE_COST[cell] for cell in route_cells[route_id])
        + env.route_distance_bias[route_id]
        for route_id in range(len(ROUTES))
    )
    candidate_pairs = tuple(
        (route_id, action)
        for route_id in range(len(ROUTES))
        for action in _action_space(route_id=route_id, phase=phase)
    )
    return GenerationContext(
        route_cells=route_cells,
        route_base_cost=route_base_cost,
        candidate_pairs=candidate_pairs,
    )


class RuleEngine:
    """Adaptive rule penalty manager."""

    def __init__(self, rules: tuple[RuleDefinition, ...]) -> None:
        self.rules = rules
        self.penalties: dict[str, float] = {rule.name: rule.penalty for rule in rules}
        self.rule_map: dict[str, RuleDefinition] = {rule.name: rule for rule in rules}

    def evaluate_break_flags(
        self,
        *,
        energy_spent: float,
        recharge_gained: float,
        verify_attempt: bool,
    ) -> dict[str, int]:
        flags: dict[str, int] = {name: 0 for name in self.penalties}
        if "late_delivery" in flags and energy_spent > 9.5:
            flags["late_delivery"] = 1
        if "skip_verification" in flags and not verify_attempt:
            flags["skip_verification"] = 1
        if "low_inventory" in flags and recharge_gained <= 0.2:
            flags["low_inventory"] = 1
        return flags

    def penalty_cost(self, break_flags: dict[str, int]) -> float:
        return sum(self.penalties[name] * broke for name, broke in break_flags.items())

    def break_rates(self, counts: dict[str, int], population_size: int) -> dict[str, float]:
        return {
            name: (counts.get(name, 0) / max(population_size, 1))
            for name in self.penalties
        }

    def update(self, break_rates: dict[str, float]) -> None:
        for name, current_penalty in list(self.penalties.items()):
            definition = self.rule_map[name]
            updated = current_penalty + definition.alpha * (
                break_rates[name] - definition.target_rate
            )
            self.penalties[name] = _clip(
                updated, definition.min_penalty, definition.max_penalty
            )


class LayeredState:
    """3-layer grid state with cross-layer couplings."""

    def __init__(self) -> None:
        self.layer_risk: dict[str, dict[int, float]] = {
            LAYER_PHYSICAL: {
                0: 4.2,
                1: 3.6,
                2: 4.2,
                3: 3.6,
                4: 2.8,
                5: 3.6,
                6: 4.2,
                7: 3.6,
                8: 4.2,
            },
            LAYER_INFO: {cell: 1.6 for cell in range(9)},
            LAYER_SOCIAL: {cell: 1.1 for cell in range(9)},
        }
        self.neutrality: dict[int, float] = {cell: 0.5 for cell in range(9)}
        self.pheromone: dict[int, float] = {cell: 0.0 for cell in range(9)}
        self.teacher_spike: dict[int, float] = {cell: 0.0 for cell in range(9)}
        self.route_distance_bias: dict[int, float] = {route_id: 0.0 for route_id in range(8)}
        self.route_risk_bias: dict[int, float] = {route_id: 0.0 for route_id in range(8)}

    def configure_for_phase(self, phase: PhaseConfig) -> None:
        for route_id in range(8):
            self.route_distance_bias[route_id] = 0.0
            self.route_risk_bias[route_id] = 0.0
        if phase.dual_niche:
            self.route_distance_bias[1] = 1.8
            self.route_risk_bias[1] = -1.2
            self.route_risk_bias[4] = 1.6

    def route_cells(self, route_id: int) -> tuple[int, int, int]:
        return ROUTES[route_id]

    def flicker_risk(self, generation: int, config: ManifoldConfig) -> float:
        cycle = (generation // config.flicker_period) % 2
        return config.flicker_low if cycle == 0 else config.flicker_high

    def recharge_multiplier(self, cell: int, generation: int, config: ManifoldConfig) -> float:
        if cell != 4:
            return 1.0
        cycle = (generation // config.recharge_flicker_period) % 2
        return config.recharge_flicker_low if cycle == 0 else config.recharge_flicker_high

    def layer_cell_risk(
        self,
        *,
        layer: str,
        cell: int,
        route_id: int,
        generation: int,
        phase: PhaseConfig,
        config: ManifoldConfig,
    ) -> float:
        base = self.layer_risk[layer][cell]
        if layer == LAYER_PHYSICAL:
            risk = base + self.pheromone[cell] + self.teacher_spike[cell]
            risk += self.route_risk_bias[route_id]
            if phase.flicker_enabled and route_id == config.flicker_route:
                risk = self.flicker_risk(generation=generation, config=config)
            return max(0.0, risk)
        if layer == LAYER_INFO:
            return max(0.0, base + 0.25 * self.teacher_spike[cell])
        return max(0.0, base)

    def combined_physical_risk(
        self,
        *,
        cell: int,
        route_id: int,
        generation: int,
        phase: PhaseConfig,
        config: ManifoldConfig,
    ) -> tuple[float, dict[str, float]]:
        physical = self.layer_cell_risk(
            layer=LAYER_PHYSICAL,
            cell=cell,
            route_id=route_id,
            generation=generation,
            phase=phase,
            config=config,
        )
        layer_contrib = {
            LAYER_PHYSICAL: physical,
            LAYER_INFO: 0.0,
            LAYER_SOCIAL: 0.0,
        }
        if phase.multi_layer_enabled:
            info = self.layer_cell_risk(
                layer=LAYER_INFO,
                cell=cell,
                route_id=route_id,
                generation=generation,
                phase=phase,
                config=config,
            )
            social = self.layer_cell_risk(
                layer=LAYER_SOCIAL,
                cell=cell,
                route_id=route_id,
                generation=generation,
                phase=phase,
                config=config,
            )
            propagated_info = config.layer_coupling_info_to_physical * info
            propagated_social = config.layer_coupling_social_to_info * social
            layer_contrib[LAYER_INFO] = propagated_info
            layer_contrib[LAYER_SOCIAL] = propagated_social
            physical = physical + propagated_info + propagated_social
        return max(0.0, physical), layer_contrib

    def recharge_gain(
        self,
        *,
        cell: int,
        generation: int,
        phase: PhaseConfig,
        config: ManifoldConfig,
    ) -> float:
        if not phase.recharge_enabled:
            return 0.0
        if cell not in RECHARGE_CELLS:
            return 0.0
        return RECHARGE_CELLS[cell] * self.recharge_multiplier(
            cell=cell,
            generation=generation,
            config=config,
        )

    def apply_connector_event(self, event: ConnectorEvent) -> None:
        if event.layer == "physical_risk":
            self.layer_risk[LAYER_PHYSICAL][event.cell] += event.delta
        elif event.layer == "info_noise":
            self.layer_risk[LAYER_INFO][event.cell] += event.delta
        elif event.layer == "social_reputation":
            self.layer_risk[LAYER_SOCIAL][event.cell] += event.delta

    def decay(self, config: ManifoldConfig) -> None:
        for cell in range(9):
            self.pheromone[cell] *= config.pheromone_decay
            self.teacher_spike[cell] *= config.spike_decay
            self.layer_risk[LAYER_INFO][cell] *= 0.985
            self.layer_risk[LAYER_SOCIAL][cell] *= 0.992

    def deposit_death_pheromone(self, route_id: int, config: ManifoldConfig) -> None:
        for cell in self.route_cells(route_id):
            self.pheromone[cell] += config.pheromone_deposit

    def export_transfer_map(self) -> dict[int, float]:
        return {cell: _clip(value, 0.0, 1.0) for cell, value in self.neutrality.items()}


def _seed_population(config: ManifoldConfig, rng: random.Random) -> list[AgentGenome]:
    population: list[AgentGenome] = []
    for idx in range(config.initial_population):
        ratio = (idx + 0.5) / config.initial_population
        mirrored = 1.0 - ratio
        risk_multiplier = _clip(0.1 + 2.4 * ratio + rng.gauss(0.0, 0.04), 0.1, 2.5)
        max_risk = _clip(2.0 + 7.5 * mirrored + rng.gauss(0.0, 0.16), 2.0, 9.5)
        timing_bias = _clip(rng.random(), 0.0, 1.0)
        recharge_bias = _clip(rng.random(), 0.0, 1.0)
        honesty_bias = _clip(rng.random(), 0.0, 1.0)
        verification_skill = _clip(rng.random(), 0.0, 1.0)
        population.append(
            AgentGenome(
                id=idx,
                risk_multiplier=risk_multiplier,
                max_risk=max_risk,
                timing_bias=timing_bias,
                recharge_bias=recharge_bias,
                honesty_bias=honesty_bias,
                verification_skill=verification_skill,
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
    return entropy / math.log(3)


def _perception_noise(agent: AgentGenome, config: ManifoldConfig) -> float:
    return config.perception_noise_base / (1.0 + agent.age / config.perception_noise_age_scale)


def _route_has_recharge(route_id: int) -> bool:
    return any(cell in RECHARGE_CELLS for cell in ROUTES[route_id])


def _action_space(route_id: int, phase: PhaseConfig) -> tuple[str, ...]:
    if not phase.recharge_enabled:
        return (ACTION_ADVANCE,)
    actions = [ACTION_ADVANCE, ACTION_DETOUR_RECHARGE]
    if _route_has_recharge(route_id):
        actions.append(ACTION_PAUSE_RECHARGE)
    return tuple(actions)


def _initial_energy(agent: AgentGenome, phase: PhaseConfig) -> float:
    if not phase.ontogeny_enabled:
        return 0.0
    if phase.energy_budget is None:
        return agent.energy_max
    return min(agent.energy_max, phase.energy_budget)


def _predator_adjustment(
    *,
    max_reputation: float,
    current_spawn_rate: float,
    phase: PhaseConfig,
) -> float:
    if not phase.predator_auto_tuning:
        return current_spawn_rate
    if max_reputation > 0.85:
        return _clip(current_spawn_rate + 0.02, 0.0, 0.4)
    if max_reputation < 0.70:
        return _clip(current_spawn_rate - 0.01, 0.0, 0.4)
    return current_spawn_rate


def _compute_memory_revenue(
    *,
    agent: AgentGenome,
    verify_cost: float,
    phase: PhaseConfig,
) -> float:
    if not phase.memory_market_enabled:
        return 0.0
    p_lying = _clip(1.0 - (0.55 * agent.honesty_bias + 0.45 * agent.reputation), 0.0, 0.95)
    ceiling = 0.81
    reputation_effective = min(agent.reputation, ceiling)
    inventory_multiplier = 1.0 + min(agent.verified_memory, 10.0) / 10.0
    return verify_cost * (1.0 - p_lying) * (0.5 + reputation_effective) * inventory_multiplier


def _estimate_verify_attempt(
    *,
    agent: AgentGenome,
    layer_info_noise: float,
    rng: random.Random,
) -> bool:
    threshold = 0.28 + 0.18 * layer_info_noise
    propensity = (
        0.60 * agent.verification_skill
        + 0.30 * agent.honesty_bias
        + 0.10 * agent.reputation
        + rng.random() * 0.12
    )
    opportunistic_verify = (
        agent.verification_skill + agent.honesty_bias >= 1.2 and layer_info_noise <= 2.6
    )
    return propensity >= threshold or opportunistic_verify


def _build_explain_log(
    *,
    agent: AgentGenome,
    cell: int,
    left: float,
    verify_cost: float,
    info_risk: float,
    chose_verify: bool,
) -> str:
    relation = "<" if left < verify_cost * info_risk else ">="
    decision = "verified" if chose_verify else "skipped verification"
    return (
        f"Agent {agent.id} {decision} at cell {cell} because "
        f"{left:.3f} {relation} {verify_cost:.2f}×{info_risk:.2f}"
    )


def _compute_path_cost(
    agent: AgentGenome,
    route_id: int,
    action: str,
    env: LayeredState,
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
    rule_engine: RuleEngine,
    predator_spawn_rate: float,
    *,
    route_cells: tuple[int, int, int] | None = None,
    route_base_cost: float | None = None,
    estimate_only: bool,
    rng: random.Random,
) -> RouteOutcome:
    cached_cells = route_cells if route_cells is not None else env.route_cells(route_id)
    base_cost = (
        route_base_cost
        if route_base_cost is not None
        else sum(CELL_BASE_COST[cell] for cell in cached_cells) + env.route_distance_bias[route_id]
    )

    risk_cost = 0.0
    energy_spent = 0.0
    recharge_gained = 0.0
    memory_revenue = 0.0
    rule_penalty_cost = 0.0
    energy_remaining = _initial_energy(agent=agent, phase=phase)
    died = False
    noise = _perception_noise(agent=agent, config=config)
    did_pause_recharge = False
    verify_happened = False
    layer_costs = {layer: 0.0 for layer in LAYER_NAMES}
    explain_log = ""

    if phase.recharge_enabled and action == ACTION_DETOUR_RECHARGE:
        base_cost += config.detour_penalty + 0.35 * (1.0 - agent.timing_bias)
        detour_gain = config.recharge_reward_scale * (5.0 + 4.0 * agent.recharge_bias)
        new_energy = min(agent.energy_max, energy_remaining + detour_gain)
        recharge_gained += new_energy - energy_remaining
        energy_remaining = new_energy

    for cell in cached_cells:
        combined_risk, layer_contrib = env.combined_physical_risk(
            cell=cell,
            route_id=route_id,
            generation=generation,
            phase=phase,
            config=config,
        )
        info_noise = env.layer_cell_risk(
            layer=LAYER_INFO,
            cell=cell,
            route_id=route_id,
            generation=generation,
            phase=phase,
            config=config,
        )
        used_risk = combined_risk
        if estimate_only:
            used_risk = max(0.0, combined_risk + rng.gauss(0.0, noise))

        verify_attempt = _estimate_verify_attempt(
            agent=agent,
            layer_info_noise=info_noise,
            rng=rng,
        )
        verify_cost = 0.08 * (1.0 + info_noise)
        if verify_attempt:
            base_cost += verify_cost
            memory_revenue += _compute_memory_revenue(
                agent=agent,
                verify_cost=verify_cost,
                phase=phase,
            )
            verify_happened = True
        explain_log = _build_explain_log(
            agent=agent,
            cell=cell,
            left=used_risk * 0.01,
            verify_cost=verify_cost,
            info_risk=info_noise,
            chose_verify=verify_attempt,
        )

        if phase.ontogeny_enabled:
            proactive_fraction = _clip(
                0.12 + 0.45 * agent.timing_bias + 0.14 * agent.risk_multiplier,
                0.0,
                0.85,
            )
            if action == ACTION_DETOUR_RECHARGE:
                proactive_fraction = _clip(proactive_fraction + 0.08, 0.0, 0.9)
            required_armor = max(0.0, used_risk - agent.max_risk)
            target_residual = max(0.0, used_risk * (1.0 - proactive_fraction))
            desired_armor = max(0.0, used_risk - target_residual)
            armor_budget_need = max(required_armor, desired_armor)
            armor_cap = energy_remaining / agent.energy_per_armor
            armor_applied = min(armor_budget_need, armor_cap)
            spent = armor_applied * agent.energy_per_armor
            energy_spent += spent
            energy_remaining -= spent
        else:
            armor_applied = 0.0

        residual_risk = used_risk - armor_applied
        layer_costs[LAYER_PHYSICAL] += layer_contrib[LAYER_PHYSICAL] * agent.risk_multiplier
        layer_costs[LAYER_INFO] += layer_contrib[LAYER_INFO] * agent.risk_multiplier
        layer_costs[LAYER_SOCIAL] += layer_contrib[LAYER_SOCIAL] * agent.risk_multiplier
        risk_cost += residual_risk * agent.risk_multiplier
        if residual_risk > agent.max_risk:
            died = True
            break

        if (
            phase.recharge_enabled
            and action == ACTION_PAUSE_RECHARGE
            and not did_pause_recharge
            and cell in RECHARGE_CELLS
        ):
            pause_gain = env.recharge_gain(
                cell=cell,
                generation=generation,
                phase=phase,
                config=config,
            )
            pause_gain *= config.recharge_reward_scale * (0.55 + agent.recharge_bias)
            new_energy = min(agent.energy_max, energy_remaining + pause_gain)
            recharge_gained += new_energy - energy_remaining
            energy_remaining = new_energy
            base_cost += config.pause_penalty
            did_pause_recharge = True

    timing_adjustment = 0.0
    if phase.flicker_enabled and route_id == config.flicker_route:
        flicker_is_high = env.flicker_risk(generation=generation, config=config) > (
            config.flicker_low + config.flicker_high
        ) / 2
        timing_adjustment = 2.2 * agent.timing_bias if flicker_is_high else -1.2 * agent.timing_bias

    predator_cost = 0.0
    if phase.predator_auto_tuning:
        predator_cost = predator_spawn_rate * (1.0 + (1.0 - agent.reputation) * 2.0) * 3.0

    break_flags = rule_engine.evaluate_break_flags(
        energy_spent=energy_spent,
        recharge_gained=recharge_gained,
        verify_attempt=verify_happened,
    )
    if phase.adaptive_rules_enabled:
        rule_penalty_cost = rule_engine.penalty_cost(break_flags)
    else:
        rule_penalty_cost = 0.0
        break_flags = {name: 0 for name in break_flags}

    total_cost = (
        base_cost
        + risk_cost
        + timing_adjustment
        + rule_penalty_cost
        + predator_cost
        - memory_revenue
    )
    if phase.ontogeny_enabled:
        total_cost += energy_spent
    if phase.recharge_enabled and action == ACTION_PAUSE_RECHARGE and not did_pause_recharge:
        total_cost += 4.0
    if died:
        total_cost += config.death_penalty

    if phase.rule_targets_enabled and not estimate_only:
        top_cell = max(
            range(9),
            key=lambda cell_idx: env.combined_physical_risk(
                cell=cell_idx,
                route_id=route_id,
                generation=generation,
                phase=phase,
                config=config,
            )[0],
        )
        total_cost += 0.25 * (top_cell / 8.0)

    return RouteOutcome(
        route_id=route_id,
        action=action,
        expected_cost=total_cost if estimate_only else 0.0,
        actual_cost=total_cost if not estimate_only else 0.0,
        risk_cost=risk_cost,
        base_cost=base_cost,
        energy_spent=energy_spent,
        recharge_gained=recharge_gained,
        memory_revenue=memory_revenue,
        rule_penalty_cost=rule_penalty_cost,
        layer_costs=layer_costs,
        break_flags=break_flags,
        died=died,
        explain_log=explain_log,
    )


def _evaluate_agent(
    agent: AgentGenome,
    env: LayeredState,
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
    rule_engine: RuleEngine,
    predator_spawn_rate: float,
    rng: random.Random,
) -> tuple[float, RouteOutcome, float]:
    estimated_outcomes: list[RouteOutcome] = []
    for route_id in range(len(ROUTES)):
        for action in _action_space(route_id=route_id, phase=phase):
            estimated_outcomes.append(
                _compute_path_cost(
                    agent=agent,
                    route_id=route_id,
                    action=action,
                    env=env,
                    generation=generation,
                    phase=phase,
                    config=config,
                    rule_engine=rule_engine,
                    predator_spawn_rate=predator_spawn_rate,
                    estimate_only=True,
                    rng=rng,
                )
            )

    chosen = min(estimated_outcomes, key=lambda outcome: outcome.expected_cost)
    actual_chosen = _compute_path_cost(
        agent=agent,
        route_id=chosen.route_id,
        action=chosen.action,
        env=env,
        generation=generation,
        phase=phase,
        config=config,
        rule_engine=rule_engine,
        predator_spawn_rate=predator_spawn_rate,
        estimate_only=False,
        rng=rng,
    )

    optimal_actual_cost = actual_chosen.actual_cost
    for route_id in range(len(ROUTES)):
        for action in _action_space(route_id=route_id, phase=phase):
            if route_id == chosen.route_id and action == chosen.action:
                continue
            candidate = _compute_path_cost(
                agent=agent,
                route_id=route_id,
                action=action,
                env=env,
                generation=generation,
                phase=phase,
                config=config,
                rule_engine=rule_engine,
                predator_spawn_rate=predator_spawn_rate,
                estimate_only=False,
                rng=rng,
            )
            optimal_actual_cost = min(optimal_actual_cost, candidate.actual_cost)

    regret = max(0.0, actual_chosen.actual_cost - optimal_actual_cost)
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
        recharge_bias=_clip(parent.recharge_bias + rng.gauss(0.0, sigma * 1.6), 0.0, 1.0),
        honesty_bias=_clip(parent.honesty_bias + rng.gauss(0.0, sigma * 1.4), 0.0, 1.0),
        verification_skill=_clip(parent.verification_skill + rng.gauss(0.0, sigma * 1.5), 0.0, 1.0),
        reputation=_clip(parent.reputation + rng.gauss(0.0, sigma * 0.8), 0.0, 1.0),
        verified_memory=max(0.0, parent.verified_memory * 0.96),
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
    env: LayeredState,
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
        dominant_route = max(route_usage, key=route_usage.get)
        spike = rng.uniform(config.teacher_spike_low, config.teacher_spike_high)
        for cell in env.route_cells(dominant_route):
            env.teacher_spike[cell] += spike
            env.layer_risk[LAYER_INFO][cell] += spike * 0.2
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
                recharge_bias=elite.recharge_bias,
                honesty_bias=elite.honesty_bias,
                verification_skill=elite.verification_skill,
                reputation=elite.reputation,
                verified_memory=elite.verified_memory,
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


def _build_confidence_distribution(route_usage: dict[int, int]) -> dict[int, float]:
    cell_weights: dict[int, float] = {cell: 0.0 for cell in range(9)}
    total = sum(route_usage.values())
    if total <= 0:
        return cell_weights
    for route_id, count in route_usage.items():
        share = count / total
        cells = ROUTES[route_id]
        for cell in cells:
            cell_weights[cell] += share / len(cells)
    norm = sum(cell_weights.values())
    if norm <= 0:
        return cell_weights
    return {cell: weight / norm for cell, weight in cell_weights.items()}


def _load_rules(config: ManifoldConfig) -> tuple[RuleDefinition, ...]:
    return compile_rulebook(config.rulebook_text)


def _load_connector_events(config: ManifoldConfig) -> tuple[ConnectorEvent, ...]:
    path = Path(config.connector_events_path) if config.connector_events_path else None
    return load_connector_events(path)


def _events_for_generation_indexed(
    indexed_events: dict[int, tuple[ConnectorEvent, ...]],
    generation: int,
) -> tuple[ConnectorEvent, ...]:
    return indexed_events.get(generation, ())


def _events_for_generation(
    events: tuple[ConnectorEvent, ...],
    generation: int,
) -> tuple[ConnectorEvent, ...]:
    return tuple(event for event in events if event.generation == generation)


def _evaluate_agent_cached(
    agent: AgentGenome,
    env: LayeredState,
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
    rule_engine: RuleEngine,
    predator_spawn_rate: float,
    generation_cache: GenerationContext,
    rng: random.Random,
) -> tuple[float, RouteOutcome, float]:
    estimated_outcomes: list[RouteOutcome] = []
    for route_id, action in generation_cache.candidate_pairs:
        estimated_outcomes.append(
            _compute_path_cost(
                agent=agent,
                route_id=route_id,
                action=action,
                env=env,
                generation=generation,
                phase=phase,
                config=config,
                rule_engine=rule_engine,
                predator_spawn_rate=predator_spawn_rate,
                route_cells=generation_cache.route_cells[route_id],
                route_base_cost=generation_cache.route_base_cost[route_id],
                estimate_only=True,
                rng=rng,
            )
        )

    chosen = min(estimated_outcomes, key=lambda outcome: outcome.expected_cost)
    actual_chosen = _compute_path_cost(
        agent=agent,
        route_id=chosen.route_id,
        action=chosen.action,
        env=env,
        generation=generation,
        phase=phase,
        config=config,
        rule_engine=rule_engine,
        predator_spawn_rate=predator_spawn_rate,
        route_cells=generation_cache.route_cells[chosen.route_id],
        route_base_cost=generation_cache.route_base_cost[chosen.route_id],
        estimate_only=False,
        rng=rng,
    )

    optimal_actual_cost = actual_chosen.actual_cost
    for route_id, action in generation_cache.candidate_pairs:
        if route_id == chosen.route_id and action == chosen.action:
            continue
        candidate = _compute_path_cost(
            agent=agent,
            route_id=route_id,
            action=action,
            env=env,
            generation=generation,
            phase=phase,
            config=config,
            rule_engine=rule_engine,
            predator_spawn_rate=predator_spawn_rate,
            route_cells=generation_cache.route_cells[route_id],
            route_base_cost=generation_cache.route_base_cost[route_id],
            estimate_only=False,
            rng=rng,
        )
        optimal_actual_cost = min(optimal_actual_cost, candidate.actual_cost)

    regret = max(0.0, actual_chosen.actual_cost - optimal_actual_cost)
    return regret, actual_chosen, optimal_actual_cost


def _evaluate_agent_vectorized(
    agent: AgentGenome,
    env: LayeredState,
    generation: int,
    phase: PhaseConfig,
    config: ManifoldConfig,
    rule_engine: RuleEngine,
    predator_spawn_rate: float,
    generation_cache: GenerationContext,
    rng: random.Random,
) -> tuple[float, RouteOutcome, float]:
    # Uses cached candidate list to reduce Python-level construction overhead.
    return _evaluate_agent_cached(
        agent=agent,
        env=env,
        generation=generation,
        phase=phase,
        config=config,
        rule_engine=rule_engine,
        predator_spawn_rate=predator_spawn_rate,
        generation_cache=generation_cache,
        rng=rng,
    )


def run_manifold(
    config: ManifoldConfig | None = None,
    *,
    transfer_neutrality_layer: dict[int, float] | None = None,
) -> SimulationResult:
    """Run full MANIFOLD simulation and return all telemetry."""

    if config is None:
        config = ManifoldConfig()
    rng = random.Random(config.seed)
    env = LayeredState()
    if transfer_neutrality_layer is not None:
        for cell, value in transfer_neutrality_layer.items():
            if cell in env.neutrality:
                env.neutrality[cell] = _clip(float(value), 0.0, 1.0)
    population = _seed_population(config=config, rng=rng)
    rules = _load_rules(config=config)
    rule_engine = RuleEngine(rules=rules)
    connector_events = _load_connector_events(config=config)
    indexed_events = (
        _index_events_by_generation(connector_events)
        if config.use_event_indexing
        else {}
    )
    predator_spawn_rate = config.predator_spawn_rate

    metrics: list[GenerationMetrics] = []
    id_counter = max(agent.id for agent in population) + 1
    absolute_generation = 0

    for phase in config.phases:
        env.configure_for_phase(phase)
        for _ in range(phase.generations):
            events_for_generation = (
                _events_for_generation_indexed(indexed_events, absolute_generation)
                if config.use_event_indexing
                else _events_for_generation(connector_events, absolute_generation)
            )
            for event in events_for_generation:
                env.apply_connector_event(event)

            generation_cache = (
                _prepare_generation_context(env=env, phase=phase)
                if config.use_generation_cache
                else None
            )

            regrets: dict[int, float] = {}
            route_usage: dict[int, int] = {route_id: 0 for route_id in range(len(ROUTES))}
            action_counts: dict[str, int] = {
                ACTION_ADVANCE: 0,
                ACTION_PAUSE_RECHARGE: 0,
                ACTION_DETOUR_RECHARGE: 0,
            }
            rule_break_counts: dict[str, int] = {rule.name: 0 for rule in rules}
            layer_accumulator: dict[str, float] = {layer: 0.0 for layer in LAYER_NAMES}
            deaths = 0
            total_energy = 0.0
            total_recharge = 0.0
            total_memory_revenue = 0.0
            recharge_events = 0
            explain_samples: list[str] = []

            for agent in population:
                if generation_cache is not None and config.use_vectorized_scoring:
                    regret, outcome, _ = _evaluate_agent_vectorized(
                        agent=agent,
                        env=env,
                        generation=absolute_generation,
                        phase=phase,
                        config=config,
                        rule_engine=rule_engine,
                        predator_spawn_rate=predator_spawn_rate,
                        generation_cache=generation_cache,
                        rng=rng,
                    )
                elif generation_cache is not None:
                    regret, outcome, _ = _evaluate_agent_cached(
                        agent=agent,
                        env=env,
                        generation=absolute_generation,
                        phase=phase,
                        config=config,
                        rule_engine=rule_engine,
                        predator_spawn_rate=predator_spawn_rate,
                        generation_cache=generation_cache,
                        rng=rng,
                    )
                else:
                    regret, outcome, _ = _evaluate_agent(
                        agent=agent,
                        env=env,
                        generation=absolute_generation,
                        phase=phase,
                        config=config,
                        rule_engine=rule_engine,
                        predator_spawn_rate=predator_spawn_rate,
                        rng=rng,
                    )
                regrets[agent.id] = regret
                route_usage[outcome.route_id] += 1
                action_counts[outcome.action] += 1
                for layer in LAYER_NAMES:
                    layer_accumulator[layer] += outcome.layer_costs.get(layer, 0.0)
                for rule_name, broke in outcome.break_flags.items():
                    rule_break_counts[rule_name] += broke

                agent.last_route = outcome.route_id
                agent.last_action = outcome.action
                agent.last_energy_spent = outcome.energy_spent
                agent.last_recharge_gained = outcome.recharge_gained
                agent.last_memory_revenue = outcome.memory_revenue
                agent.last_died = outcome.died
                agent.last_explanation = outcome.explain_log
                if outcome.explain_log and len(explain_samples) < 6:
                    explain_samples.append(outcome.explain_log)

                total_energy += outcome.energy_spent
                total_recharge += outcome.recharge_gained
                total_memory_revenue += outcome.memory_revenue
                if outcome.recharge_gained > 0.0:
                    recharge_events += 1
                if outcome.died:
                    deaths += 1
                    env.deposit_death_pheromone(route_id=outcome.route_id, config=config)

                agent.verified_memory = _clip(
                    agent.verified_memory + (1.0 if "verified" in outcome.explain_log else -0.2),
                    0.0,
                    30.0,
                )
                agent.reputation = _clip(
                    agent.reputation + (0.012 if "verified" in outcome.explain_log else -0.01),
                    0.0,
                    1.0,
                )

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
            avg_recharge = total_recharge / len(population)
            avg_memory_revenue = total_memory_revenue / len(population)
            recharge_event_rate = recharge_events / len(population)
            death_rate = deaths / len(population)
            max_reputation = max(agent.reputation for agent in population)

            rule_break_rates = rule_engine.break_rates(
                counts=rule_break_counts,
                population_size=len(population),
            )
            if phase.adaptive_rules_enabled:
                rule_engine.update(rule_break_rates)

            if phase.predator_auto_tuning:
                predator_spawn_rate = _predator_adjustment(
                    max_reputation=max_reputation,
                    current_spawn_rate=predator_spawn_rate,
                    phase=phase,
                )

            confidence_distribution = (
                _build_confidence_distribution(route_usage)
                if phase.rule_targets_enabled
                else {}
            )
            if phase.multi_layer_enabled:
                for cell, confidence in confidence_distribution.items():
                    env.neutrality[cell] = _clip(
                        env.neutrality[cell] * 0.90 + confidence * 0.10,
                        0.0,
                        1.0,
                    )

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

            layer_scale = max(len(population), 1)
            layer_regret = {
                layer: layer_accumulator[layer] / layer_scale for layer in LAYER_NAMES
            }
            metrics.append(
                GenerationMetrics(
                    generation=absolute_generation,
                    phase=phase.name,
                    population_size=len(population),
                    avg_regret=avg_regret,
                    best_regret=best_regret,
                    diversity=diversity,
                    average_energy_spent=avg_energy,
                    average_recharge_gained=avg_recharge,
                    average_memory_revenue=avg_memory_revenue,
                    recharge_event_rate=recharge_event_rate,
                    death_rate=death_rate,
                    predator_spawn_rate=predator_spawn_rate,
                    max_reputation=max_reputation,
                    layer_regret_contrib=layer_regret,
                    rule_break_rates=rule_break_rates,
                    rule_penalties=dict(rule_engine.penalties),
                    teacher_event=teacher_event,
                    niche_counts=niche_counts,
                    route_usage=route_usage,
                    action_counts=action_counts,
                    confidence_distribution=confidence_distribution,
                    target_intent=config.target_intent,
                    explain_samples=explain_samples,
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

    transfer_artifact = {
        "neutrality_layer": env.export_transfer_map(),
        "rule_penalties": dict(rule_engine.penalties),
        "max_reputation": max(agent.reputation for agent in population),
    }
    return SimulationResult(
        config=config,
        metrics=metrics,
        final_population=population,
        transfer_artifact=transfer_artifact,
    )


def summarize_result(result: SimulationResult) -> dict[str, Any]:
    """Compact summary for CLI/UI usage."""

    final = result.metrics[-1]
    by_phase: dict[str, dict[str, float | str | dict[str, float]]] = {}
    for phase in result.config.phases:
        phase_metrics = [m for m in result.metrics if m.phase == phase.name]
        end = phase_metrics[-1]
        dominant_action = max(end.action_counts, key=end.action_counts.get)
        by_phase[phase.name] = {
            "avg_regret_start": phase_metrics[0].avg_regret,
            "avg_regret_end": end.avg_regret,
            "diversity_end": end.diversity,
            "energy_end": end.average_energy_spent,
            "recharge_end": end.average_recharge_gained,
            "memory_revenue_end": end.average_memory_revenue,
            "recharge_event_rate_end": end.recharge_event_rate,
            "death_rate_end": end.death_rate,
            "dominant_action_end": dominant_action,
            "predator_spawn_rate_end": end.predator_spawn_rate,
            "layer_regret_contrib_end": end.layer_regret_contrib,
        }
    return {
        "total_generations": result.config.total_generations,
        "final_population": len(result.final_population),
        "final_avg_regret": final.avg_regret,
        "final_best_regret": final.best_regret,
        "final_diversity": final.diversity,
        "final_energy_spent": final.average_energy_spent,
        "final_recharge_gained": final.average_recharge_gained,
        "final_memory_revenue": final.average_memory_revenue,
        "final_recharge_event_rate": final.recharge_event_rate,
        "final_action_mix": final.action_counts,
        "final_rule_penalties": final.rule_penalties,
        "final_layer_regret": final.layer_regret_contrib,
        "transfer_artifact": result.transfer_artifact,
        "phase_summary": by_phase,
    }
