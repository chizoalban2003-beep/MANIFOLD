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


# ---------------------------------------------------------------------------
# GridState — ready-to-optimize grid produced by DynamicTranslator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridState:
    """A fully configured :class:`GridWorld` ready for physics-engine processing.

    Produced by :meth:`DynamicTranslator.map_problem`.  Carry the configured
    world alongside the metadata used to produce it so callers can inspect the
    mapping and reproduce it deterministically.

    Attributes
    ----------
    world:
        The ``GridWorld`` with cells, targets, and rules already populated.
    description:
        The original problem description string.
    domain:
        Inferred or supplied domain label (e.g. ``"finance"``).
    parameters:
        The normalised parameter dict used to build the grid cells.
    cell_profile:
        A summary ``[cost, risk, neutrality, asset]`` tuple representing the
        *centre cell* of the grid — useful for quick inspection.
    """

    world: "GridWorld"
    description: str
    domain: str
    parameters: dict[str, object]
    cell_profile: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# DynamicTranslator — maps arbitrary problems into the 4-vector grid
# ---------------------------------------------------------------------------


@dataclass
class DynamicTranslator:
    """Translate any arbitrary problem into a MANIFOLD 31×31 grid environment.

    :class:`DynamicTranslator` is the "Universal Grid Translator" described in
    the MANIFOLD architecture.  It converts a free-text *description* plus a
    *parameters* dict into a fully populated :class:`GridWorld` that the
    physics engine (:class:`AgentPopulation`) can immediately optimise.

    The mapping is **deterministic**: the same ``(description, parameters)``
    pair always produces the same grid.  Determinism is guaranteed by deriving
    the random seed from the description string rather than from wall-clock
    time.

    Parameters
    ----------
    grid_size:
        Side length of the square grid.  Default: ``31``.
    n_targets:
        Number of dynamic targets (profit nodes) to inject.  Default: ``3``.
    infer_domain:
        When ``True`` (default), the translator examines *description* for
        known domain keywords and uses them to adjust the risk/asset profile
        if the ``"domain"`` key is absent from *parameters*.

    Example
    -------
    ::

        translator = DynamicTranslator()
        state = translator.map_problem(
            description="API rate limit management",
            parameters={"cost": 0.25, "uncertainty": 0.6, "stakes": 0.5},
        )
        result = AgentPopulation().optimize(state.world)
    """

    grid_size: int = 31
    n_targets: int = 3
    infer_domain: bool = True

    # ------------------------------------------------------------------
    # Domain-archetype profiles: (cost_bias, risk_bias, asset_bias)
    # ------------------------------------------------------------------

    _DOMAIN_PROFILES: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "finance":    (0.20, 0.35, 0.80),
            "legal":      (0.18, 0.30, 0.75),
            "medical":    (0.15, 0.40, 0.85),
            "research":   (0.12, 0.20, 0.65),
            "creative":   (0.08, 0.10, 0.60),
            "support":    (0.10, 0.15, 0.70),
            "security":   (0.25, 0.50, 0.80),
            "logistics":  (0.20, 0.25, 0.70),
            "compute":    (0.30, 0.20, 0.65),
            "general":    (0.15, 0.20, 0.65),
        },
        init=False,
        repr=False,
    )

    # Keywords used to infer domain from the description string
    _DOMAIN_KEYWORDS: dict[str, list[str]] = field(
        default_factory=lambda: {
            "finance":   ["financ", "payment", "invoice", "bank", "trade", "ledger", "accounting"],
            "legal":     ["legal", "contract", "compliance", "regulation", "law", "clause"],
            "medical":   ["medical", "clinical", "patient", "diagnosis", "health", "drug"],
            "research":  ["research", "experiment", "hypothesis", "analysis", "science", "study"],
            "creative":  ["creative", "design", "art", "content", "story", "marketing"],
            "support":   ["support", "ticket", "triage", "customer", "helpdesk", "crm"],
            "security":  ["security", "threat", "intrusion", "attack", "vulnerab", "audit"],
            "logistics": ["logistics", "delivery", "route", "fleet", "shipping", "supply"],
            "compute":   ["compute", "api", "rate limit", "throughput", "latency", "server"],
        },
        init=False,
        repr=False,
    )

    def map_problem(
        self,
        description: str,
        parameters: dict[str, object],
    ) -> GridState:
        """Map an arbitrary problem description and parameters into a grid state.

        The mapping is fully deterministic: calling this method twice with
        identical arguments always returns an equivalent :class:`GridState`.

        Parameters
        ----------
        description:
            Free-text description of the problem domain (e.g.
            ``"API rate limit management"``).
        parameters:
            A flat dict of named problem dimensions.  Recognised keys:

            * ``"cost"`` — base cell cost [0, 1].  Default: 0.15.
            * ``"risk"`` — base cell risk [0, 1].  Default: 0.20.
            * ``"uncertainty"`` — elevates both risk and neutrality [0, 1].
            * ``"stakes"`` — scales asset value [0, 1].
            * ``"complexity"`` — increases cost and lowers neutrality [0, 1].
            * ``"asset"`` — direct asset override [0, 1].
            * ``"domain"`` — explicit domain label (skips keyword inference).
            * ``"n_targets"`` — override for :attr:`n_targets`.
            * ``"moves"`` — target move pattern: ``"static"``, ``"random_walk"``,
              or ``"cycle"``.  Default: ``"static"``.

        Returns
        -------
        GridState
            A fully configured world ready for :class:`AgentPopulation`.
        """
        # ------------------------------------------------------------------
        # 1. Derive a deterministic seed from the description
        # ------------------------------------------------------------------
        seed = self._description_seed(description)
        rng = random.Random(seed)

        # ------------------------------------------------------------------
        # 2. Normalise parameters
        # ------------------------------------------------------------------
        p = self._normalise(parameters)

        # ------------------------------------------------------------------
        # 3. Infer / look up domain archetype
        # ------------------------------------------------------------------
        domain = str(p.get("domain", ""))
        if not domain and self.infer_domain:
            domain = self._infer_domain(description)
        if not domain:
            domain = "general"
        cost_bias, risk_bias, asset_bias = self._DOMAIN_PROFILES.get(
            domain, self._DOMAIN_PROFILES["general"]
        )

        # ------------------------------------------------------------------
        # 4. Build per-cell coefficients
        # ------------------------------------------------------------------
        base_cost = float(p.get("cost", cost_bias))
        base_risk = float(p.get("risk", risk_bias))
        base_asset = float(p.get("asset", asset_bias))
        uncertainty = float(p.get("uncertainty", 0.30))
        stakes = float(p.get("stakes", 0.50))
        complexity = float(p.get("complexity", 0.40))

        # Effective risk is elevated by uncertainty
        eff_risk = _clamp(base_risk + 0.25 * uncertainty)
        # Effective cost is elevated by complexity
        eff_cost = _clamp(base_cost + 0.15 * complexity)
        # Neutrality decreases with complexity and stakes
        eff_neutrality = _clamp(0.90 - 0.35 * complexity - 0.20 * stakes)
        # Asset scales with stakes and the domain asset bias
        eff_asset = _clamp(base_asset * (0.60 + 0.80 * stakes))

        # ------------------------------------------------------------------
        # 5. Populate GridWorld
        # ------------------------------------------------------------------
        n_targets = int(p.get("n_targets", self.n_targets))
        moves = str(p.get("moves", "static"))

        world = GridWorld(size=self.grid_size, seed=seed)
        center = self.grid_size // 2
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                dist = (abs(row - center) + abs(col - center)) / max(1, self.grid_size)
                cell_cost = _clamp(eff_cost + 0.08 * dist + rng.uniform(-0.03, 0.03))
                cell_risk = _clamp(eff_risk + 0.10 * dist + rng.uniform(-0.04, 0.04))
                cell_neutrality = _clamp(eff_neutrality - 0.05 * dist)
                cell_asset = _clamp(eff_asset * max(0.1, 1.0 - dist * 1.5))
                world.set_cell(row, col, cell_cost, cell_risk, cell_neutrality, cell_asset)

        # Targets distributed across the grid deterministically
        for i in range(n_targets):
            target_row = rng.randint(2, self.grid_size - 3)
            target_col = rng.randint(2, self.grid_size - 3)
            target_asset = _clamp(eff_asset * (1.5 + 0.5 * i))
            world.add_dynamic_targets(
                [
                    {
                        "id": f"target_{i}",
                        "pos": (target_row, target_col),
                        "asset": target_asset,
                        "moves": moves,
                    }
                ]
            )

        # Rules derived from the risk / stakes profile
        world.add_rule("miss_goal", _clamp(1.0 + stakes * 5.0), "miss_target")
        world.add_rule("deception_penalty", _clamp(1.0 + eff_risk * 3.0), "deception_detected")
        if eff_risk > 0.35:
            world.add_rule("trusted_lie_penalty", _clamp(1.0 + eff_risk * 4.0), "trusted_lie")

        # Centre-cell profile for inspection
        cx_cell = world.cells[center][center]
        cell_profile = (cx_cell.cost, cx_cell.risk, cx_cell.neutrality, cx_cell.asset)

        return GridState(
            world=world,
            description=description,
            domain=domain,
            parameters=dict(p),
            cell_profile=cell_profile,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _description_seed(self, description: str) -> int:
        """Derive a deterministic integer seed from the description string."""
        h = 5381
        for ch in description.lower():
            h = ((h << 5) + h) ^ ord(ch)
        return abs(h) % (2 ** 31)

    def _infer_domain(self, description: str) -> str:
        """Return the best-matching domain keyword from *description*, or ''."""
        desc_lower = description.lower()
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                return domain
        return ""

    @staticmethod
    def _normalise(parameters: dict[str, object]) -> dict[str, object]:
        """Return a copy of *parameters* with float values clamped to [0, 1]."""
        float_keys = {
            "cost", "risk", "asset", "uncertainty",
            "stakes", "complexity",
        }
        result: dict[str, object] = {}
        for key, val in parameters.items():
            if key in float_keys:
                try:
                    result[key] = _clamp(float(val))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    result[key] = val
            else:
                result[key] = val
        return result


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Return *value* clamped to [*low*, *high*]."""
    return max(low, min(high, value))


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
