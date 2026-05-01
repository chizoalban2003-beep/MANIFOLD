"""Reusable GridMapper OS layer for Project MANIFOLD.

GridMapper turns a domain problem into a MANIFOLD world:

- each cell is a `[cost, risk, neutrality, asset]` vector,
- targets can be static, moving, or multiple,
- rules are priced penalties,
- a gen-2000 social population optimizes the resulting substrate.

The implementation intentionally reuses the social MANIFOLD engine rather than
forking new agent logic. The wrapper compiles dynamic targets into the grid at
each generation, then lets the evolved trust/verification ecology adapt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import csv
import random
from statistics import fmean
from typing import Iterable, Literal

from .social import (
    CellVector,
    PolicyAudit,
    SocialConfig,
    SocialGenerationSummary,
    SocialGenome,
    SocialManifoldExperiment,
    compile_policy_audit,
    load_grid_from_csv,
    seed_social_population,
)


Position = tuple[int, int]
MovePattern = Literal["static", "random_walk", "cycle"]


@dataclass
class DynamicTarget:
    """A goal or profit node that can move or change value over time."""

    id: str
    pos: Position
    asset: float
    moves: MovePattern = "static"
    path: tuple[Position, ...] = ()

    def position_at(self, tick: int, grid_size: int, rng: random.Random) -> Position:
        if self.moves == "static":
            return self.pos
        if self.moves == "cycle" and self.path:
            return self.path[tick % len(self.path)]
        if self.moves == "random_walk":
            row, col = self.pos
            stable_id = sum((index + 1) * ord(char) for index, char in enumerate(self.id))
            local = random.Random(stable_id + tick + rng.randrange(10_000))
            for _ in range(tick + 1):
                dr, dc = local.choice(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))
                row = max(0, min(grid_size - 1, row + dr))
                col = max(0, min(grid_size - 1, col + dc))
            return (row, col)
        return self.pos


@dataclass(frozen=True)
class Rule:
    """A priced constraint in the problem world."""

    name: str
    penalty: float
    triggers: str


@dataclass
class GridOptimizationResult:
    """Output of a GridMapper OS optimization run."""

    history: list[SocialGenerationSummary]
    audit: PolicyAudit
    target_snapshots: dict[int, tuple[Position, ...]]
    rules: tuple[Rule, ...]
    rule_penalty_budget: float

    @property
    def verification(self) -> float:
        return self.audit.recommended_verification_rate

    @property
    def gossip(self) -> float:
        return self.audit.recommended_gossip_rate

    @property
    def reputation_cap(self) -> float:
        return self.audit.recommended_predation_threshold

    @property
    def max_reputation(self) -> float:
        return max(0.0, min(1.0, self.audit.recommended_predation_threshold))


@dataclass
class GridWorld:
    """Problem compiler from domain cells/targets/rules to MANIFOLD substrate."""

    size: int = 31
    default_cell: CellVector = field(
        default_factory=lambda: CellVector(cost=0.05, risk=0.05, neutrality=0.95, asset=0.0)
    )
    cells: list[list[CellVector]] = field(init=False)
    targets: list[DynamicTarget] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    seed: int = 2500

    def __post_init__(self) -> None:
        self.cells = [[self.default_cell for _ in range(self.size)] for _ in range(self.size)]

    @classmethod
    def from_csv(cls, path: str, size: int = 31) -> "GridWorld":
        world = cls(size=size)
        world.cells = load_grid_from_csv(path, size)
        return world

    def load_from_csv(self, path: str) -> None:
        self.cells = load_grid_from_csv(path, self.size)

    def load_from_traffic_csv(self, path: str) -> None:
        """Alias for domain-specific examples."""

        self.load_from_csv(path)

    def set_cell(
        self,
        row: int,
        col: int,
        cost: float,
        risk: float,
        neutrality: float = 0.0,
        asset: float = 0.0,
    ) -> None:
        self._validate_position((row, col))
        self.cells[row][col] = CellVector(cost=cost, risk=risk, neutrality=neutrality, asset=asset)

    def add_dynamic_targets(self, targets: Iterable[dict | DynamicTarget]) -> None:
        for target in targets:
            if isinstance(target, DynamicTarget):
                self.targets.append(target)
            else:
                self.targets.append(
                    DynamicTarget(
                        id=str(target["id"]),
                        pos=tuple(target["pos"]),  # type: ignore[arg-type]
                        asset=float(target["asset"]),
                        moves=target.get("moves", "static"),
                        path=tuple(tuple(item) for item in target.get("path", ())),
                    )
                )

    def add_rule(self, name: str, penalty: float, triggers: str) -> None:
        self.rules.append(Rule(name=name, penalty=penalty, triggers=triggers))

    def compiled_grid(self, tick: int) -> list[list[CellVector]]:
        rng = random.Random(self.seed + tick)
        grid = [[cell for cell in row] for row in self.cells]
        for target in self.targets:
            row, col = target.position_at(tick, self.size, rng)
            base = grid[row][col]
            grid[row][col] = CellVector(
                cost=base.cost,
                risk=base.risk,
                neutrality=base.neutrality,
                asset=base.asset + target.asset,
            )
        return grid

    def target_positions(self, tick: int) -> tuple[Position, ...]:
        rng = random.Random(self.seed + tick)
        return tuple(target.position_at(tick, self.size, rng) for target in self.targets)

    def export_heatmap_csv(self, path: str, tick: int = 0) -> None:
        """Write a simple asset-risk heat map for external visualization."""

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("row", "col", "cost", "risk", "neutrality", "asset"),
            )
            writer.writeheader()
            for row_idx, row in enumerate(self.compiled_grid(tick)):
                for col_idx, cell in enumerate(row):
                    writer.writerow(
                        {
                            "row": row_idx,
                            "col": col_idx,
                            "cost": cell.cost,
                            "risk": cell.risk,
                            "neutrality": cell.neutrality,
                            "asset": cell.asset,
                        }
                    )

    def _validate_position(self, position: Position) -> None:
        row, col = position
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position {position} is outside {self.size}x{self.size} grid.")


@dataclass
class AgentPopulation:
    """High-level optimizer seeded with evolved MANIFOLD social DNA."""

    seed: str = "gen2000"
    n: int = 200
    predators: float = 0.05
    random_seed: int = 2500

    def optimize(
        self,
        world: GridWorld,
        generations: int = 300,
        verification_cost: float = 0.30,
        false_trust_penalty: float = 0.50,
    ) -> GridOptimizationResult:
        config = SocialConfig(
            population_size=self.n,
            generations=generations,
            grid_size=world.size,
            seed=self._seed_value(),
            verification_cost=verification_cost,
            false_trust_penalty=false_trust_penalty,
        )
        experiment = SocialManifoldExperiment(config)
        experiment.population = self._seed_population(config)

        history: list[SocialGenerationSummary] = []
        target_snapshots: dict[int, tuple[Position, ...]] = {}
        for generation in range(generations):
            experiment.grid = world.compiled_grid(generation)
            target_snapshots[generation] = world.target_positions(generation)
            experiment.source_counts = [0 for _ in experiment.population]
            results = [
                experiment._evaluate_agent(index, genome, generation)
                for index, genome in enumerate(experiment.population)
            ]
            results = self._apply_rule_penalties(results, world)
            experiment._update_reputation(results)
            experiment.population = experiment._reproduce(results)
            experiment.reputation = [0.0 for _ in experiment.population]
            history.append(experiment._summarize(generation, results))

        audit = compile_policy_audit(history, config)
        return GridOptimizationResult(
            history=history,
            audit=audit,
            target_snapshots=target_snapshots,
            rules=tuple(world.rules),
            rule_penalty_budget=fmean(rule.penalty for rule in world.rules) if world.rules else 0.0,
        )

    def _apply_rule_penalties(
        self, results: list, world: GridWorld
    ) -> list:
        if not world.rules:
            return results

        adjusted = []
        for result in results:
            penalty = 0.0
            for rule in world.rules:
                if rule.triggers == "miss_target" and result.assets_collected <= 0:
                    penalty += rule.penalty
                elif rule.triggers == "deception_detected" and result.lies_detected > 0:
                    penalty += rule.penalty * result.lies_detected
                elif rule.triggers == "trusted_lie" and result.trusted_lies > 0:
                    penalty += rule.penalty * result.trusted_lies
                elif rule.triggers == "low_energy" and result.energy_left < 1.0:
                    penalty += rule.penalty
            if penalty:
                adjusted.append(
                    replace(
                        result,
                        fitness=result.fitness - penalty,
                        penalties=result.penalties + penalty,
                    )
                )
            else:
                adjusted.append(result)
        return adjusted

    def _seed_value(self) -> int:
        if self.seed == "gen2000":
            return 2500
        try:
            return int(self.seed)
        except ValueError:
            return self.random_seed

    def _seed_population(self, config: SocialConfig) -> list[SocialGenome]:
        rng = random.Random(config.seed)
        population = seed_social_population(self.n, rng, config)
        predator_count = min(self.n, max(0, round(self.n * self.predators)))
        for index in range(predator_count):
            population[index] = SocialGenome(
                deception=0.15,
                verification=0.85,
                gossip=0.20,
                memory=0.45,
                energy=10.0,
                predation_threshold=0.85,
                ancestry="predatory-scout",
            )
        return population
