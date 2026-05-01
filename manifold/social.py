"""Social-intelligence engine for Project MANIFOLD.

This module models the later MANIFOLD claim directly: do not program social
rules such as honesty, verification, gossip, blacklisting, or mercy. Price the
actions, let agents compete for assets on a `[cost, risk, neutrality, asset]`
grid, and select the genomes that preserve the most energy and value.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
import random
from statistics import fmean
from typing import Iterable, Literal


Position = tuple[int, int]
PresetName = Literal["trust", "birmingham", "misinformation", "compute", "collusion"]


@dataclass(frozen=True)
class CellVector:
    """Universal mapper cell: cost, risk, neutrality, asset."""

    cost: float
    risk: float
    neutrality: float
    asset: float


@dataclass(frozen=True)
class SocialConfig:
    """Configuration for social-rule evolution."""

    population_size: int = 180
    generations: int = 500
    grid_size: int = 31
    seed: int = 2500
    preset: PresetName = "trust"
    life_steps: int = 36
    signal_cost: float = 0.10
    verification_cost: float = 0.30
    verification_risk_reduction: float = 0.70
    gossip_cost: float = 0.10
    false_trust_penalty: float = 0.50
    detected_lie_penalty: float = 0.65
    blacklist_after_lies: int = 3
    survivor_fraction: float = 0.20
    mutation_sigma: float = 0.045
    reputation_cap: float = 1.5
    random_audit_rate: float = 0.08
    source_share_limit: float = 0.35
    monopoly_penalty: float = 0.20
    predatory_scouts: bool = True
    scout_source_share_trigger: float = 0.22
    scout_reputation_trigger: float = 0.75
    scout_verification_discount: float = 0.35
    data_path: str | None = None


@dataclass
class SocialGenome:
    """Five-number DNA for social MANIFOLD agents."""

    deception: float
    verification: float
    gossip: float
    memory: float
    energy: float
    predation_threshold: float = 0.85
    ancestry: str = "seed"

    def mutated(self, rng: random.Random, config: SocialConfig) -> "SocialGenome":
        def clamp(value: float, low: float, high: float) -> float:
            return max(low, min(high, value))

        return SocialGenome(
            deception=clamp(self.deception + rng.gauss(0.0, config.mutation_sigma), 0.0, 1.0),
            verification=clamp(self.verification + rng.gauss(0.0, config.mutation_sigma), 0.0, 1.0),
            gossip=clamp(self.gossip + rng.gauss(0.0, config.mutation_sigma), 0.0, 1.0),
            memory=clamp(self.memory + rng.gauss(0.0, config.mutation_sigma), 0.0, 1.0),
            energy=clamp(self.energy + rng.gauss(0.0, config.mutation_sigma * 8.0), 4.0, 20.0),
            predation_threshold=clamp(
                self.predation_threshold + rng.gauss(0.0, config.mutation_sigma), 0.55, 0.98
            ),
            ancestry=self.niche(),
        )

    def niche(self) -> str:
        if self.verification >= 0.60 and self.gossip <= 0.40 and self.deception <= 0.35:
            return "Scout"
        if self.verification >= 0.70 and self.gossip >= 0.55:
            return "Verifier"
        if self.deception >= 0.45:
            return "Deceiver"
        if self.gossip >= 0.65:
            return "Gossip"
        return "Pragmatist"

    @property
    def memory_ticks(self) -> int:
        return int(5 + self.memory * 55)


@dataclass(frozen=True)
class SocialLifeResult:
    genome: SocialGenome
    fitness: float
    energy_left: float
    assets_collected: float
    penalties: float
    signals_sent: int
    lies_sent: int
    verifications: int
    lies_detected: int
    trusted_lies: int
    predatory_scout_checks: int
    gossip_events: int
    blacklist_events: int
    forgiveness_events: int
    niche: str


@dataclass(frozen=True)
class SocialGenerationSummary:
    generation: int
    average_fitness: float
    best_fitness: float
    average_deception: float
    average_verification: float
    average_gossip: float
    average_memory_ticks: float
    average_start_energy: float
    average_predation_threshold: float
    lie_rate: float
    verification_rate: float
    gossip_rate: float
    trusted_lie_rate: float
    predatory_scout_rate: float
    blacklist_rate: float
    forgiveness_rate: float
    honest_correlation: float
    diversity: float
    top_source_share: float
    source_hhi: float
    monopoly_pressure: float
    niche_counts: dict[str, int]


@dataclass(frozen=True)
class PolicyAudit:
    """Post-game policy recommendations compiled from an evolved run."""

    verification_threshold: float
    recommended_verification_rate: float
    recommended_gossip_rate: float
    recommended_predation_threshold: float
    recommended_blacklist_after_lies: int
    recommended_forgiveness_window: int
    expected_deception_equilibrium: float
    robustness_score: float
    monopoly_risk: float
    monopoly_controls: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass
class SocialManifoldExperiment:
    """Evolutionary social-rule experiment on a universal vector grid."""

    config: SocialConfig = field(default_factory=SocialConfig)
    rng: random.Random = field(init=False)
    grid: list[list[CellVector]] = field(init=False)
    population: list[SocialGenome] = field(init=False)
    reputation: list[float] = field(init=False)
    source_counts: list[int] = field(init=False)
    history: list[SocialGenerationSummary] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.grid = (
            load_grid_from_csv(self.config.data_path, self.config.grid_size)
            if self.config.data_path
            else build_problem_grid(self.config)
        )
        self.population = seed_social_population(self.config.population_size, self.rng, self.config)
        self.reputation = [0.0 for _ in self.population]
        self.source_counts = [0 for _ in self.population]

    def run(self) -> list[SocialGenerationSummary]:
        for generation in range(self.config.generations):
            results = self.step(generation)
            self.history.append(self._summarize(generation, results))
        return self.history

    def step(self, generation: int) -> list[SocialLifeResult]:
        self.source_counts = [0 for _ in self.population]
        results = [self._evaluate_agent(index, genome, generation) for index, genome in enumerate(self.population)]
        self._update_reputation(results)
        self.population = self._reproduce(results)
        self.reputation = [0.0 for _ in self.population]
        return results

    def _evaluate_agent(
        self, agent_index: int, genome: SocialGenome, generation: int
    ) -> SocialLifeResult:
        position = (self.config.grid_size // 2, self.config.grid_size // 2)
        energy = genome.energy
        assets = 0.0
        penalties = 0.0
        signals_sent = 0
        lies_sent = 0
        verifications = 0
        lies_detected = 0
        trusted_lies = 0
        predatory_scout_checks = 0
        gossip_events = 0
        blacklist_events = 0
        forgiveness_events = 0
        betrayals: dict[int, int] = {}
        blacklist_until: dict[int, int] = {}

        for tick in range(self.config.life_steps):
            blacklist_until = {
                sender: until for sender, until in blacklist_until.items() if until > tick
            }
            if tick and tick % max(1, genome.memory_ticks) == 0 and betrayals:
                forgive_count = len(betrayals)
                betrayals.clear()
                forgiveness_events += forgive_count

            options = list(neighbors(position, self.config.grid_size))
            sender_index = self._choose_sender(agent_index, blacklist_until)
            suggested = self._best_neighbor(options)
            signal_is_lie = False
            if sender_index is not None:
                sender = self.population[sender_index]
                signals_sent += 1
                if self.rng.random() < sender.deception:
                    suggested = self._worst_neighbor(options)
                    signal_is_lie = True
                    lies_sent += 1

            use_signal = sender_index is not None and sender_index not in blacklist_until
            scout_check = self._should_predatory_scout_check(genome, sender_index)
            if use_signal and (scout_check or self.rng.random() < genome.verification):
                verifications += 1
                if scout_check:
                    predatory_scout_checks += 1
                    energy -= self.config.verification_cost * self.config.scout_verification_discount
                else:
                    energy -= self.config.verification_cost
                if signal_is_lie:
                    lies_detected += 1
                    penalties += self.config.detected_lie_penalty * 0.20
                    betrayals[sender_index] = betrayals.get(sender_index, 0) + 1
                    if betrayals[sender_index] >= self.config.blacklist_after_lies:
                        blacklist_until[sender_index] = tick + genome.memory_ticks
                        blacklist_events += 1
                    if self.rng.random() < genome.gossip:
                        gossip_events += self._gossip(sender_index)
                        energy -= self.config.gossip_cost
                    use_signal = False

            if use_signal:
                chosen = suggested
                if signal_is_lie:
                    cell = self.cell_at(chosen)
                    trusted_lies += 1
                    penalties += cell.risk * self.config.false_trust_penalty
            else:
                chosen = self._best_neighbor(options, risk_sensitive=True)

            cell = self.cell_at(chosen)
            position = chosen
            energy -= cell.cost
            risk_loss = cell.risk * (1.0 - self.config.verification_risk_reduction * (verifications > 0))
            penalties += risk_loss * 0.05
            assets += cell.asset
            if energy <= 0:
                penalties += abs(energy) * 0.5
                break

        fitness = energy + assets - penalties
        return SocialLifeResult(
            genome=genome,
            fitness=fitness,
            energy_left=energy,
            assets_collected=assets,
            penalties=penalties,
            signals_sent=signals_sent,
            lies_sent=lies_sent,
            verifications=verifications,
            lies_detected=lies_detected,
            trusted_lies=trusted_lies,
            predatory_scout_checks=predatory_scout_checks,
            gossip_events=gossip_events,
            blacklist_events=blacklist_events,
            forgiveness_events=forgiveness_events,
            niche=genome.niche(),
        )

    def cell_at(self, position: Position) -> CellVector:
        row, col = position
        return self.grid[row][col]

    def _choose_sender(
        self, agent_index: int, blacklist_until: dict[int, int]
    ) -> int | None:
        if len(self.population) < 2:
            return None
        candidates = [
            index
            for index in range(len(self.population))
            if index != agent_index and index not in blacklist_until
        ]
        if not candidates:
            return None
        if self.rng.random() < self.config.random_audit_rate:
            selected = self.rng.choice(candidates)
        else:
            selected = max(
                candidates,
                key=lambda index: min(self.reputation[index], self.config.reputation_cap)
                + self.rng.random() * 0.05,
            )
        self.source_counts[selected] += 1
        return selected

    def _best_neighbor(self, options: Iterable[Position], risk_sensitive: bool = False) -> Position:
        def score(position: Position) -> float:
            cell = self.cell_at(position)
            risk_weight = 1.4 if risk_sensitive else 0.8
            return cell.asset - cell.cost - risk_weight * cell.risk + 0.15 * cell.neutrality

        return max(options, key=score)

    def _worst_neighbor(self, options: Iterable[Position]) -> Position:
        def trap_score(position: Position) -> float:
            cell = self.cell_at(position)
            return cell.risk + cell.cost - cell.asset

        return max(options, key=trap_score)

    def _should_predatory_scout_check(
        self, genome: SocialGenome, sender_index: int | None
    ) -> bool:
        if (
            not self.config.predatory_scouts
            or sender_index is None
            or genome.niche() != "Scout"
        ):
            return False
        total_sources = sum(self.source_counts) or 1
        source_share = self.source_counts[sender_index] / total_sources
        reputation = min(self.reputation[sender_index], self.config.reputation_cap)
        return (
            source_share >= self.config.scout_source_share_trigger
            or reputation >= genome.predation_threshold
        )

    def _gossip(self, sender_index: int) -> int:
        radius = 5
        warnings = min(radius, max(0, len(self.reputation) - 1))
        self.reputation[sender_index] -= 0.25 * warnings
        return warnings

    def _update_reputation(self, results: list[SocialLifeResult]) -> None:
        total_sources = sum(self.source_counts) or 1
        for index, result in enumerate(results):
            self.reputation[index] += result.lies_detected * 0.08
            self.reputation[index] -= result.lies_sent * 0.12
            self.reputation[index] += result.gossip_events * 0.01
            source_share = self.source_counts[index] / total_sources
            if source_share > self.config.source_share_limit:
                excess = source_share - self.config.source_share_limit
                self.reputation[index] -= excess * self.config.monopoly_penalty * len(results)

    def _reproduce(self, results: list[SocialLifeResult]) -> list[SocialGenome]:
        ordered = sorted(results, key=lambda result: result.fitness, reverse=True)
        survivor_count = max(2, int(len(ordered) * self.config.survivor_fraction))
        survivors = [result.genome for result in ordered[:survivor_count]]
        next_population = list(survivors)
        while len(next_population) < self.config.population_size:
            parent = self.rng.choice(survivors)
            next_population.append(parent.mutated(self.rng, self.config))
        return next_population

    def _summarize(
        self, generation: int, results: list[SocialLifeResult]
    ) -> SocialGenerationSummary:
        total_signals = sum(result.signals_sent for result in results) or 1
        total_lies = sum(result.lies_sent for result in results)
        total_verifications = sum(result.verifications for result in results)
        total_gossip = sum(result.gossip_events for result in results)
        niche_counts = {
            niche: sum(result.niche == niche for result in results)
            for niche in ("Scout", "Verifier", "Deceiver", "Gossip", "Pragmatist")
        }
        honest_labels = [1 - int(result.lies_sent > 0) for result in results]
        fitness_labels = [1 if result.fitness >= fmean(r.fitness for r in results) else 0 for result in results]
        total_sources = sum(self.source_counts) or 1
        source_shares = [count / total_sources for count in self.source_counts]
        top_source_share = max(source_shares, default=0.0)
        source_hhi = sum(share * share for share in source_shares)
        return SocialGenerationSummary(
            generation=generation,
            average_fitness=fmean(result.fitness for result in results),
            best_fitness=max(result.fitness for result in results),
            average_deception=fmean(result.genome.deception for result in results),
            average_verification=fmean(result.genome.verification for result in results),
            average_gossip=fmean(result.genome.gossip for result in results),
            average_memory_ticks=fmean(result.genome.memory_ticks for result in results),
            average_start_energy=fmean(result.genome.energy for result in results),
            average_predation_threshold=fmean(
                result.genome.predation_threshold for result in results
            ),
            lie_rate=total_lies / total_signals,
            verification_rate=total_verifications / total_signals,
            gossip_rate=total_gossip / total_signals,
            trusted_lie_rate=sum(result.trusted_lies for result in results) / total_signals,
            predatory_scout_rate=sum(result.predatory_scout_checks for result in results)
            / total_signals,
            blacklist_rate=sum(result.blacklist_events for result in results) / len(results),
            forgiveness_rate=sum(result.forgiveness_events for result in results) / len(results),
            honest_correlation=binary_correlation(honest_labels, fitness_labels),
            diversity=social_diversity(result.genome for result in results),
            top_source_share=top_source_share,
            source_hhi=source_hhi,
            monopoly_pressure=max(0.0, top_source_share - self.config.source_share_limit),
            niche_counts=niche_counts,
        )


def build_problem_grid(config: SocialConfig) -> list[list[CellVector]]:
    if config.data_path:
        return load_grid_from_csv(config.data_path, config.grid_size)

    rng = random.Random(config.seed + 10_001)
    grid: list[list[CellVector]] = []
    center = config.grid_size // 2
    for row in range(config.grid_size):
        grid_row = []
        for col in range(config.grid_size):
            dist = abs(row - center) + abs(col - center)
            asset_peak = max(0.0, 1.0 - dist / max(1, config.grid_size))
            noise = rng.random() * 0.08
            if config.preset == "birmingham":
                cost = 0.04 + 0.018 * dist + noise
                risk = 0.12 + 0.22 * traffic_band(row, col, config.grid_size)
                asset = 0.08 + asset_peak * 0.55 + order_cluster(row, col, config.grid_size)
            elif config.preset == "misinformation":
                cost = 0.015 + noise * 0.1
                risk = 0.35 + 0.45 * asset_peak + noise
                asset = 0.20 + 0.35 * asset_peak
            elif config.preset == "compute":
                cost = 0.05 + 0.03 * server_heat(row, col, config.grid_size)
                risk = 0.08 + 0.62 * server_heat(row, col, config.grid_size)
                asset = 0.30 + 0.55 * asset_peak
            else:
                cost = 0.05 + 0.02 * dist / max(1, config.grid_size) + noise
                risk = 0.10 + 0.55 * asset_peak + noise
                asset = 0.10 + 0.60 * asset_peak
            neutrality = max(0.0, 1.0 - min(1.0, cost + risk + asset) / 2.0)
            grid_row.append(CellVector(cost=cost, risk=risk, neutrality=neutrality, asset=asset))
        grid.append(grid_row)
    return grid


def load_grid_from_csv(path: str, grid_size: int) -> list[list[CellVector]]:
    """Load a real-world mapper CSV into a square MANIFOLD grid.

    Required columns are `row`, `col`, `cost`, `risk`, `asset`. `neutrality` is
    optional and is derived when absent. Missing cells become neutral low-value
    states, which lets sparse traffic or supply-chain extracts still run.
    """

    grid = [
        [CellVector(cost=0.05, risk=0.05, neutrality=0.95, asset=0.0) for _ in range(grid_size)]
        for _ in range(grid_size)
    ]
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"row", "col", "cost", "risk", "asset"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV grid is missing required columns: {sorted(missing)}")
        for record in reader:
            row = int(record["row"])
            col = int(record["col"])
            if not (0 <= row < grid_size and 0 <= col < grid_size):
                continue
            cost = clamp(float(record["cost"]), 0.0, 10.0)
            risk = clamp(float(record["risk"]), 0.0, 10.0)
            asset = clamp(float(record["asset"]), 0.0, 10.0)
            if record.get("neutrality") not in (None, ""):
                neutrality = clamp(float(record["neutrality"]), 0.0, 10.0)
            else:
                neutrality = max(0.0, 1.0 - min(1.0, cost + risk + asset) / 2.0)
            grid[row][col] = CellVector(
                cost=cost,
                risk=risk,
                neutrality=neutrality,
                asset=asset,
            )
    return grid


def seed_social_population(
    size: int, rng: random.Random, config: SocialConfig
) -> list[SocialGenome]:
    population: list[SocialGenome] = []
    for _ in range(size):
        if config.seed >= 2000:
            deception = clamp(rng.gauss(0.32, 0.08), 0.0, 1.0)
            verification = clamp(rng.gauss(0.54, 0.10), 0.0, 1.0)
            gossip = clamp(rng.gauss(0.67, 0.10), 0.0, 1.0)
            memory = clamp(rng.gauss(0.45, 0.12), 0.0, 1.0)
            energy = clamp(rng.gauss(10.0, 1.2), 4.0, 20.0)
            predation_threshold = clamp(rng.gauss(0.85, 0.06), 0.55, 0.98)
        else:
            deception = rng.uniform(0.05, 0.15)
            verification = rng.uniform(0.05, 0.25)
            gossip = rng.uniform(0.05, 0.30)
            memory = rng.uniform(0.10, 0.50)
            energy = rng.uniform(8.0, 12.0)
            predation_threshold = rng.uniform(0.70, 0.95)
        population.append(
            SocialGenome(
                deception,
                verification,
                gossip,
                memory,
                energy,
                predation_threshold,
            )
        )
    return population


def recommended_prices(preset: PresetName) -> dict[str, float]:
    """Domain price presets used by CLI, dashboard, and docs."""

    if preset == "birmingham":
        return {
            "signal_cost": 0.10,
            "verification_cost": 0.18,
            "false_trust_penalty": 0.75,
            "detected_lie_penalty": 0.60,
        }
    if preset == "misinformation":
        return {
            "signal_cost": 0.01,
            "verification_cost": 0.015,
            "false_trust_penalty": 0.50,
            "detected_lie_penalty": 0.85,
        }
    if preset == "compute":
        return {
            "signal_cost": 0.02,
            "verification_cost": 0.02,
            "false_trust_penalty": 1.50,
            "detected_lie_penalty": 1.00,
        }
    return {
        "signal_cost": 0.10,
        "verification_cost": 0.30,
        "false_trust_penalty": 0.50,
        "detected_lie_penalty": 0.65,
    }


def config_for_preset(
    preset: PresetName, generations: int = 500, population_size: int = 180, seed: int = 2500
) -> SocialConfig:
    prices = recommended_prices(preset)
    return SocialConfig(
        generations=generations,
        population_size=population_size,
        seed=seed,
        preset=preset,
        **prices,
    )


def run_social_experiment(config: SocialConfig | None = None) -> list[SocialGenerationSummary]:
    experiment = SocialManifoldExperiment(config or SocialConfig())
    return experiment.run()


def compile_policy_audit(
    history: list[SocialGenerationSummary], config: SocialConfig
) -> PolicyAudit:
    """Turn a MANIFOLD run into deployable policy recommendations."""

    if not history:
        raise ValueError("Cannot compile a policy audit without generation history.")

    window = history[-min(25, len(history)) :]
    final = history[-1]
    expected_deception = fmean(item.average_deception for item in window)
    recommended_verification = fmean(item.average_verification for item in window)
    recommended_gossip = fmean(item.average_gossip for item in window)
    recommended_predation = fmean(item.average_predation_threshold for item in window)
    forgiveness_window = round(fmean(item.average_memory_ticks for item in window))
    monopoly_risk = fmean(item.monopoly_pressure for item in window)

    # Verify when expected loss beats checking cost. This is the deployable
    # threshold for any external mapper that can estimate local lie probability.
    verification_threshold = min(
        1.0,
        config.verification_cost
        / max(0.001, config.false_trust_penalty * config.verification_risk_reduction),
    )
    robustness_score = clamp(
        1.0
        - final.trusted_lie_rate
        - monopoly_risk
        + min(0.25, final.diversity * 0.20),
        0.0,
        1.0,
    )

    notes: list[str] = []
    if final.average_verification < verification_threshold:
        notes.append("Verification is under-selected for the configured lie penalty.")
    if final.lie_rate > final.verification_rate:
        notes.append("Deception pressure exceeds verification pressure; raise audit rate or lie penalties.")
    if monopoly_risk > 0.05:
        notes.append("Information-source concentration is high; cap reputation and force random audits.")
    if final.forgiveness_rate == 0.0:
        notes.append("No rehabilitation observed; lower memory duration or blacklist threshold if brittleness appears.")
    if not notes:
        notes.append("Policy prices produced a stable trust economy in the observed window.")

    return PolicyAudit(
        verification_threshold=verification_threshold,
        recommended_verification_rate=recommended_verification,
        recommended_gossip_rate=recommended_gossip,
        recommended_predation_threshold=recommended_predation,
        recommended_blacklist_after_lies=config.blacklist_after_lies,
        recommended_forgiveness_window=forgiveness_window,
        expected_deception_equilibrium=expected_deception,
        robustness_score=robustness_score,
        monopoly_risk=monopoly_risk,
        monopoly_controls=(
            "cap reputation scores",
            "force random audits",
            "activate predatory scout checks on concentrated high-reputation sources",
            "penalize source-share concentration",
            "rotate verifier selection",
        ),
        notes=tuple(notes),
    )


def neighbors(position: Position, grid_size: int) -> Iterable[Position]:
    row, col = position
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (row + dr, col + dc)
        if 0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size:
            yield nxt


def social_diversity(genomes: Iterable[SocialGenome]) -> float:
    points = [
        (genome.deception, genome.verification, genome.gossip, genome.memory, genome.energy / 20.0)
        for genome in genomes
    ]
    if len(points) < 2:
        return 0.0
    return fmean(
        math.dist(left, right)
        for index, left in enumerate(points)
        for right in points[index + 1 :]
    )


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


def traffic_band(row: int, col: int, grid_size: int) -> float:
    hubs = ((grid_size // 2, grid_size // 2), (grid_size // 3, grid_size // 3), (2 * grid_size // 3, grid_size // 3))
    return max(0.0, 1.0 - min(abs(row - hr) + abs(col - hc) for hr, hc in hubs) / grid_size)


def order_cluster(row: int, col: int, grid_size: int) -> float:
    hubs = ((grid_size // 2, grid_size // 2), (grid_size // 2 + 4, grid_size // 2 - 3), (grid_size // 2 - 5, grid_size // 2 + 5))
    return 0.35 * max(0.0, 1.0 - min(abs(row - hr) + abs(col - hc) for hr, hc in hubs) / 12.0)


def server_heat(row: int, col: int, grid_size: int) -> float:
    edge = min(row, col, grid_size - row - 1, grid_size - col - 1)
    return max(0.0, 1.0 - edge / max(1, grid_size // 2))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
