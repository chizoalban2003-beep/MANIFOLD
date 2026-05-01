from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Route:
    name: str
    cells: tuple[int, int, int]
    length: float
    corridor: str


@dataclass
class ScenarioConfig:
    name: str
    scout_risk: float
    tank_risk: float
    hybrid_risk: float
    flicker_low: float
    flicker_high: float
    flicker_period: int
    teacher_enabled: bool
    energy_enabled: bool
    targeted_teacher_prob: float
    teacher_interval: int
    plateau_tolerance: float
    congestion_weight: float
    death_pheromone: float
    pheromone_decay: float
    seed_population_min: int
    seed_population_max: int


@dataclass
class Spike:
    cell: int
    intensity: float
    ttl: int
    source: str


@dataclass
class AgentGenome:
    risk_multiplier: float
    max_risk: float
    energy_policy: float
    armor_efficiency: float
    learning_rate: float
    explore_rate: float
    route_bias: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    energy_max: float = 30.0

    def copy(self) -> "AgentGenome":
        return AgentGenome(
            risk_multiplier=float(self.risk_multiplier),
            max_risk=float(self.max_risk),
            energy_policy=float(self.energy_policy),
            armor_efficiency=float(self.armor_efficiency),
            learning_rate=float(self.learning_rate),
            explore_rate=float(self.explore_rate),
            route_bias=self.route_bias.copy(),
            energy_max=float(self.energy_max),
        )


@dataclass
class TraversalResult:
    route_index: int
    route_name: str
    route_cells: tuple[int, int, int]
    cost: float
    energy_used: float
    death: bool
    unresolved_risk: float


def default_routes() -> List[Route]:
    return [
        Route("row_top", (0, 1, 2), 3.2, "scout"),
        Route("row_mid", (3, 4, 5), 2.5, "tank"),
        Route("row_bot", (6, 7, 8), 3.2, "scout"),
        Route("col_left", (0, 3, 6), 3.4, "hybrid"),
        Route("col_mid", (1, 4, 7), 2.7, "tank"),
        Route("col_right", (2, 5, 8), 3.4, "hybrid"),
        Route("diag_main", (0, 4, 8), 2.8, "flicker"),
        Route("diag_anti", (2, 4, 6), 2.8, "tank"),
    ]


def scenario_catalogue() -> Dict[str, ScenarioConfig]:
    return {
        "geometry": ScenarioConfig(
            name="Static geometry",
            scout_risk=0.8,
            tank_risk=0.9,
            hybrid_risk=0.9,
            flicker_low=1.0,
            flicker_high=1.0,
            flicker_period=8,
            teacher_enabled=False,
            energy_enabled=False,
            targeted_teacher_prob=0.0,
            teacher_interval=15,
            plateau_tolerance=0.02,
            congestion_weight=0.10,
            death_pheromone=0.0,
            pheromone_decay=0.90,
            seed_population_min=20,
            seed_population_max=28,
        ),
        "emergent": ScenarioConfig(
            name="Emergent adaptation",
            scout_risk=2.0,
            tank_risk=6.0,
            hybrid_risk=4.0,
            flicker_low=3.0,
            flicker_high=7.0,
            flicker_period=8,
            teacher_enabled=True,
            energy_enabled=False,
            targeted_teacher_prob=0.70,
            teacher_interval=15,
            plateau_tolerance=0.03,
            congestion_weight=0.18,
            death_pheromone=1.0,
            pheromone_decay=0.86,
            seed_population_min=24,
            seed_population_max=36,
        ),
        "ontogeny": ScenarioConfig(
            name="Ontogeny energy budget",
            scout_risk=2.0,
            tank_risk=6.0,
            hybrid_risk=4.0,
            flicker_low=3.0,
            flicker_high=7.0,
            flicker_period=8,
            teacher_enabled=True,
            energy_enabled=True,
            targeted_teacher_prob=0.70,
            teacher_interval=15,
            plateau_tolerance=0.03,
            congestion_weight=0.20,
            death_pheromone=1.2,
            pheromone_decay=0.84,
            seed_population_min=28,
            seed_population_max=40,
        ),
    }


def reshape_cells(values: List[float]) -> np.ndarray:
    return np.array(values, dtype=float).reshape(3, 3)


def extract_teacher_events(history: List[dict]) -> List[dict]:
    events = []
    for snapshot in history:
        if snapshot.get("teacher_event"):
            events.append(
                {
                    "generation": snapshot["generation"],
                    "event": snapshot["teacher_event"],
                    "dominant_niche": snapshot["dominant_niche"],
                }
            )
    return events


def summarise_results(results: dict) -> dict:
    history = results["history"]
    first = history[0]
    last = history[-1]
    teacher_events = extract_teacher_events(history)
    energy_series = [item["avg_energy_used"] for item in history]
    return {
        "scenario": results["scenario"],
        "start_avg_regret": first["avg_regret"],
        "end_avg_regret": last["avg_regret"],
        "best_regret": min(item["best_regret"] for item in history),
        "final_diversity": last["diversity"],
        "final_center_share": last["center_share"],
        "final_avg_energy": last["avg_energy_used"],
        "peak_avg_energy": max(energy_series),
        "cumulative_avg_energy": float(np.sum(energy_series)),
        "teacher_events": len(teacher_events),
        "final_population": len(results["population"]),
    }


class ManifoldSimulation:
    def __init__(
        self,
        seed: int = 11,
        scenario: str = "ontogeny",
        population_size: int | None = None,
        mutation_sigma: float = 0.05,
    ) -> None:
        scenarios = scenario_catalogue()
        if scenario not in scenarios:
            raise ValueError(f"Unknown scenario '{scenario}'.")

        self.rng = np.random.default_rng(seed)
        self.routes = default_routes()
        self.route_overlap = self._route_overlap_matrix()
        self.config = scenarios[scenario]
        self.population_size = population_size or self.rng.integers(
            self.config.seed_population_min,
            self.config.seed_population_max + 1,
        )
        self.mutation_sigma = mutation_sigma
        self.pheromone = np.zeros(9, dtype=float)
        self.spikes: List[Spike] = []
        self.previous_route_loads = np.zeros(len(self.routes), dtype=float)
        self.history: List[dict] = []
        self.population = self._seed_population(self.population_size)

    @staticmethod
    def route_membership(routes: List[Route] | None = None) -> np.ndarray:
        routes = routes or default_routes()
        counts = np.zeros(9, dtype=float)
        for route in routes:
            for cell in route.cells:
                counts[cell] += 1.0
        return counts

    def run(self, generations: int = 80) -> dict:
        for generation in range(generations):
            environment = self._environment_state(generation)
            choices = self._choose_routes(environment)
            snapshot = self._evaluate_generation(generation, environment, choices)
            self.history.append(snapshot)
            self.previous_route_loads = np.array(snapshot["route_loads"], dtype=float)
            if self.config.teacher_enabled:
                snapshot["teacher_event"] = self._maybe_apply_teacher(generation, snapshot)
            self._decay_traces()
            self.population = self._reproduce(snapshot["agents"])

        results = {
            "scenario": self.config.name,
            "config": {
                "population_size": self.population_size,
                "energy_enabled": self.config.energy_enabled,
                "teacher_enabled": self.config.teacher_enabled,
                "teacher_interval": self.config.teacher_interval,
                "targeted_teacher_prob": self.config.targeted_teacher_prob,
                "mutation_sigma": self.mutation_sigma,
            },
            "routes": [
                {
                    "name": route.name,
                    "corridor": route.corridor,
                    "cells": list(route.cells),
                    "length": route.length,
                }
                for route in self.routes
            ],
            "geometry_weights": (self.route_membership(self.routes) / len(self.routes)).tolist(),
            "history": self.history,
            "population": [self._agent_to_public(agent) for agent in self.population],
        }
        results["summary"] = summarise_results(results)
        return results

    def _seed_population(self, population_size: int) -> List[AgentGenome]:
        agents: List[AgentGenome] = []
        risk_grid = np.linspace(0.1, 2.5, population_size)
        max_risk_grid = np.linspace(2.0, 9.5, population_size)
        energy_grid = np.linspace(0.35, 1.20, population_size)
        armor_grid = np.linspace(0.8, 1.4, population_size)
        learning_grid = np.linspace(0.15, 0.85, population_size)
        explore_grid = np.linspace(0.02, 0.25, population_size)
        self.rng.shuffle(risk_grid)
        self.rng.shuffle(max_risk_grid)
        self.rng.shuffle(energy_grid)
        self.rng.shuffle(armor_grid)
        self.rng.shuffle(learning_grid)
        self.rng.shuffle(explore_grid)

        for index in range(population_size):
            agents.append(
                AgentGenome(
                    risk_multiplier=float(risk_grid[index]),
                    max_risk=float(max_risk_grid[index]),
                    energy_policy=float(energy_grid[index]),
                    armor_efficiency=float(armor_grid[index]),
                    learning_rate=float(learning_grid[index]),
                    explore_rate=float(explore_grid[index]),
                    route_bias=self.rng.normal(0.0, 0.25, len(self.routes)),
                )
            )
        return agents

    def _environment_state(self, generation: int) -> dict:
        flicker_risk = (
            self.config.flicker_low
            if (generation // self.config.flicker_period) % 2 == 0
            else self.config.flicker_high
        )
        route_risk = {
            "scout": self.config.scout_risk,
            "tank": self.config.tank_risk,
            "hybrid": self.config.hybrid_risk,
            "flicker": flicker_risk,
        }
        spike_costs = np.zeros(9, dtype=float)
        for spike in self.spikes:
            spike_costs[spike.cell] += spike.intensity
        return {
            "route_risk": route_risk,
            "spike_costs": spike_costs,
            "active_spikes": len(self.spikes),
            "flicker_risk": flicker_risk,
        }

    def _choose_routes(self, environment: dict) -> List[int]:
        choices: List[int] = []
        for agent in self.population:
            expected_costs = []
            for route_index, route in enumerate(self.routes):
                expected = self._simulate_route(
                    agent=agent,
                    route=route,
                    route_index=route_index,
                    route_load=float(self.previous_route_loads[route_index]),
                    environment=environment,
                )
                memory_bonus = 0.12 * float(agent.route_bias[route_index])
                expected_costs.append(expected.cost - memory_bonus)
            if self.rng.random() < agent.explore_rate:
                top_choices = np.argsort(expected_costs)[: min(3, len(expected_costs))]
                choices.append(int(self.rng.choice(top_choices)))
            else:
                choices.append(int(np.argmin(expected_costs)))
        return choices

    def _evaluate_generation(self, generation: int, environment: dict, choices: List[int]) -> dict:
        route_loads = np.bincount(choices, minlength=len(self.routes)).astype(float)
        cell_loads = np.zeros(9, dtype=float)
        agents_report: List[dict] = []
        niche_counts = Counter()
        deaths = 0
        costs = []
        regrets = []
        energy_used = []
        unresolved_risks = []

        for agent, route_index in zip(self.population, choices):
            route = self.routes[route_index]
            actual = self._simulate_route(
                agent=agent,
                route=route,
                route_index=route_index,
                route_load=float(route_loads[route_index]),
                environment=environment,
            )
            optimal_cost = math.inf
            optimal_route_index = route_index
            for candidate_index, candidate in enumerate(self.routes):
                candidate_result = self._simulate_route(
                    agent=agent,
                    route=candidate,
                    route_index=candidate_index,
                    route_load=float(route_loads[candidate_index]),
                    environment=environment,
                )
                if candidate_result.cost < optimal_cost:
                    optimal_cost = candidate_result.cost
                    optimal_route_index = candidate_index

            regret = max(0.0, actual.cost - optimal_cost)
            niche = self._niche_name(agent.max_risk)
            niche_counts[niche] += 1
            for cell in actual.route_cells:
                cell_loads[cell] += 1.0

            if actual.death:
                deaths += 1
                for cell in actual.route_cells:
                    self.pheromone[cell] += self.config.death_pheromone

            reward = -actual.cost
            chosen_bias = float(agent.route_bias[route_index])
            agent.route_bias[route_index] = chosen_bias + agent.learning_rate * (reward - chosen_bias) / 12.0
            for other_index in range(len(self.routes)):
                if other_index == route_index:
                    continue
                agent.route_bias[other_index] += (
                    agent.learning_rate
                    * self.route_overlap[route_index, other_index]
                    * (reward - agent.route_bias[other_index])
                    / 30.0
                )

            report = {
                "genome": agent.copy(),
                "niche": niche,
                "route_index": route_index,
                "route_name": actual.route_name,
                "route_cells": list(actual.route_cells),
                "cost": actual.cost,
                "optimal_cost": optimal_cost,
                "optimal_route_index": optimal_route_index,
                "regret": regret,
                "energy_used": actual.energy_used,
                "unresolved_risk": actual.unresolved_risk,
                "death": actual.death,
            }
            agents_report.append(report)
            costs.append(actual.cost)
            regrets.append(regret)
            energy_used.append(actual.energy_used)
            unresolved_risks.append(actual.unresolved_risk)

        diversity = self._effective_diversity(niche_counts)
        best_report = min(agents_report, key=lambda item: (item["regret"], item["cost"]))
        dominant_niche = max(niche_counts.items(), key=lambda item: item[1])[0]

        return {
            "generation": generation,
            "avg_cost": float(np.mean(costs)),
            "avg_regret": float(np.mean(regrets)),
            "best_regret": float(best_report["regret"]),
            "best_route": best_report["route_name"],
            "deaths": deaths,
            "diversity": diversity,
            "dominant_niche": dominant_niche,
            "center_share": float(cell_loads[4] / max(1.0, np.sum(cell_loads))),
            "avg_energy_used": float(np.mean(energy_used)),
            "avg_unresolved_risk": float(np.mean(unresolved_risks)),
            "active_spikes": environment["active_spikes"],
            "flicker_risk": environment["flicker_risk"],
            "route_loads": route_loads.tolist(),
            "cell_loads": cell_loads.tolist(),
            "niche_counts": dict(niche_counts),
            "agents": agents_report,
            "teacher_event": None,
        }

    def _simulate_route(
        self,
        agent: AgentGenome,
        route: Route,
        route_index: int,
        route_load: float,
        environment: dict,
    ) -> TraversalResult:
        remaining_energy = agent.energy_max if self.config.energy_enabled else 0.0
        cost = route.length + self.config.congestion_weight * max(0.0, route_load - 1.0)
        energy_used = 0.0
        unresolved_risk = 0.0
        death = False

        for cell in route.cells:
            risk = (
                1.0
                + environment["route_risk"][route.corridor]
                + environment["spike_costs"][cell]
                + self.pheromone[cell]
            )
            risk_gap = max(0.0, risk - agent.max_risk)
            spend = 0.0

            if self.config.energy_enabled and risk_gap > 0.0 and remaining_energy > 0.0:
                desired_spend = risk_gap * agent.energy_policy
                spend = min(desired_spend, remaining_energy)
                remaining_energy -= spend
                energy_used += spend

            protection = spend * agent.armor_efficiency
            unresolved = max(0.0, risk_gap - protection)
            unresolved_risk += unresolved
            step_cost = 1.0 + spend + unresolved * agent.risk_multiplier
            cost += step_cost
            if unresolved > 2.5:
                death = True
                cost += 8.0
                break

        return TraversalResult(
            route_index=route_index,
            route_name=route.name,
            route_cells=route.cells,
            cost=float(cost),
            energy_used=float(energy_used),
            death=death,
            unresolved_risk=float(unresolved_risk),
        )

    def _route_overlap_matrix(self) -> np.ndarray:
        overlap = np.zeros((len(self.routes), len(self.routes)), dtype=float)
        for i, left in enumerate(self.routes):
            left_cells = set(left.cells)
            for j, right in enumerate(self.routes):
                overlap[i, j] = len(left_cells.intersection(right.cells)) / 3.0
        return overlap

    def _effective_diversity(self, niche_counts: Counter) -> float:
        total = sum(niche_counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in niche_counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        return float(math.exp(entropy))

    def _niche_name(self, max_risk: float) -> str:
        if max_risk < 4.0:
            return "scout"
        if max_risk >= 6.0:
            return "tank"
        return "hybrid"

    def _reproduce(self, agent_reports: List[dict]) -> List[AgentGenome]:
        niche_totals = Counter(report["niche"] for report in agent_reports)
        scored = []
        for report in agent_reports:
            fitness = 1.0 / (1.0 + report["regret"])
            if report["death"]:
                fitness *= 0.15
            crowding = niche_totals[report["niche"]] / max(1, len(agent_reports))
            fitness /= 1.0 + 0.45 * crowding
            scored.append(fitness)

        fitnesses = np.array(scored, dtype=float)
        if float(np.sum(fitnesses)) <= 0.0:
            fitnesses = np.ones(len(agent_reports), dtype=float)
        probabilities = fitnesses / np.sum(fitnesses)

        elite_count = min(2, len(agent_reports))
        ranked = sorted(agent_reports, key=lambda report: (report["death"], report["regret"]))
        next_population = [ranked[index]["genome"].copy() for index in range(elite_count)]

        while len(next_population) < self.population_size:
            parent_index = int(self.rng.choice(len(agent_reports), p=probabilities))
            next_population.append(self._mutate(agent_reports[parent_index]["genome"]))
        return next_population

    def _mutate(self, parent: AgentGenome) -> AgentGenome:
        child = parent.copy()
        child.risk_multiplier = self._mutate_scalar(child.risk_multiplier, 0.1, 2.5)
        child.max_risk = self._mutate_scalar(child.max_risk, 2.0, 9.5)
        child.energy_policy = self._mutate_scalar(child.energy_policy, 0.25, 1.30)
        child.armor_efficiency = self._mutate_scalar(child.armor_efficiency, 0.7, 1.5)
        child.learning_rate = self._mutate_scalar(child.learning_rate, 0.05, 0.95)
        child.explore_rate = self._mutate_scalar(child.explore_rate, 0.01, 0.30)
        child.route_bias = child.route_bias + self.rng.normal(0.0, 0.12, len(self.routes))
        return child

    def _mutate_scalar(self, value: float, lower: float, upper: float) -> float:
        scale = (upper - lower) * self.mutation_sigma
        mutated = value + float(self.rng.normal(0.0, scale))
        return float(np.clip(mutated, lower, upper))

    def _maybe_apply_teacher(self, generation: int, snapshot: dict) -> str | None:
        if generation == 0 or generation % self.config.teacher_interval != 0:
            return None
        if len(self.history) < 2 * self.config.teacher_interval:
            return None

        recent = self.history[-self.config.teacher_interval :]
        previous = self.history[-2 * self.config.teacher_interval : -self.config.teacher_interval]
        recent_regret = np.mean([item["avg_regret"] for item in recent])
        previous_regret = np.mean([item["avg_regret"] for item in previous])
        plateau = recent_regret >= previous_regret - self.config.plateau_tolerance
        if not plateau:
            return None

        targeted = self.rng.random() < self.config.targeted_teacher_prob
        if targeted:
            niche = snapshot["dominant_niche"]
            candidate_indices = [
                index for index, route in enumerate(self.routes) if self._corridor_matches_niche(route.corridor, niche)
            ]
            if not candidate_indices:
                candidate_indices = list(range(len(self.routes)))
            chosen_index = max(candidate_indices, key=lambda index: snapshot["route_loads"][index])
            chosen_cell = int(self.rng.choice(self.routes[chosen_index].cells))
            self.spikes.append(Spike(cell=chosen_cell, intensity=2.8, ttl=4, source=f"teacher:{niche}"))
            return f"Targeted spike hit {self.routes[chosen_index].name} for the dominant {niche} niche."

        chosen_cell = int(self.rng.integers(0, 9))
        self.spikes.append(Spike(cell=chosen_cell, intensity=2.0, ttl=3, source="teacher:random"))
        return f"Random spike hit cell {chosen_cell}."

    def _corridor_matches_niche(self, corridor: str, niche: str) -> bool:
        if niche == "tank":
            return corridor in {"tank", "flicker"}
        if niche == "scout":
            return corridor == "scout"
        return corridor == "hybrid"

    def _decay_traces(self) -> None:
        self.pheromone *= self.config.pheromone_decay
        next_spikes: List[Spike] = []
        for spike in self.spikes:
            next_ttl = spike.ttl - 1
            if next_ttl > 0:
                next_spikes.append(
                    Spike(
                        cell=spike.cell,
                        intensity=spike.intensity * 0.92,
                        ttl=next_ttl,
                        source=spike.source,
                    )
                )
        self.spikes = next_spikes

    def _agent_to_public(self, agent: AgentGenome) -> dict:
        return {
            "risk_multiplier": round(agent.risk_multiplier, 3),
            "max_risk": round(agent.max_risk, 3),
            "energy_policy": round(agent.energy_policy, 3),
            "armor_efficiency": round(agent.armor_efficiency, 3),
            "learning_rate": round(agent.learning_rate, 3),
            "explore_rate": round(agent.explore_rate, 3),
            "energy_max": round(agent.energy_max, 3),
            "niche": self._niche_name(agent.max_risk),
        }
