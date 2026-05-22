"""MANIFOLD Brain: generalized adaptive decision layer for agents.

MANIFOLD Brain is the product-facing meta-controller above GridMapper OS and
TrustRouter. It chooses the next action for an AI agent, tool-using workflow, or
human-in-the-loop system by pricing the current task as cost, risk, asset, and
rule pressure.
"""

from __future__ import annotations

import logging
import uuid
import weakref

from dataclasses import dataclass, field
from typing import Any, Literal

from .cell_update_bus import CellUpdate, CellUpdateBus, get_bus
from .gridmapper import AgentPopulation, GridOptimizationResult, GridWorld
from .movement import MovementState, MovementStateMachine, PathPlanner, Watchdog
from .policy_action import PolicyAction
from .trustrouter import clamp01

_logger = logging.getLogger(__name__)


BrainAction = Literal[
    "answer",
    "clarify",
    "retrieve",
    "verify",
    "use_tool",
    "delegate",
    "plan",
    "explore",
    "exploit",
    "wait",
    "escalate",
    "refuse",
    "stop",
]


@dataclass(frozen=True)
class BrainTask:
    """A general agent task that can be mapped into MANIFOLD."""

    prompt: str
    domain: str = "general"
    uncertainty: float = 0.5
    complexity: float = 0.5
    stakes: float = 0.5
    source_confidence: float = 0.7
    tool_relevance: float = 0.5
    time_pressure: float = 0.4
    safety_sensitivity: float = 0.2
    collaboration_value: float = 0.3
    user_patience: float = 0.7
    dynamic_goal: bool = False

    def normalized(self) -> "BrainTask":
        return BrainTask(
            prompt=self.prompt,
            domain=self.domain,
            uncertainty=clamp01(self.uncertainty),
            complexity=clamp01(self.complexity),
            stakes=clamp01(self.stakes),
            source_confidence=clamp01(self.source_confidence),
            tool_relevance=clamp01(self.tool_relevance),
            time_pressure=clamp01(self.time_pressure),
            safety_sensitivity=clamp01(self.safety_sensitivity),
            collaboration_value=clamp01(self.collaboration_value),
            user_patience=clamp01(self.user_patience),
            dynamic_goal=self.dynamic_goal,
        )


@dataclass(frozen=True)
class ToolProfile:
    """A callable capability available to an agentic system."""

    name: str
    cost: float
    latency: float
    reliability: float
    risk: float
    asset: float
    domain: str = "general"

    @property
    def utility(self) -> float:
        return self.asset * self.reliability - self.cost - self.latency - self.risk


@dataclass(frozen=True)
class BrainConfig:
    grid_size: int = 11
    generations: int = 30
    population_size: int = 48
    predators: float = 0.05
    seed: int = 2500
    planning_cost: float = 0.10
    exploration_cost: float = 0.12
    delegation_cost: float = 0.25


@dataclass(frozen=True)
class BrainDecision:
    action: BrainAction
    confidence: float
    risk_score: float
    expected_utility: float
    selected_tool: str | None
    verification_rate: float
    reputation_cap: float
    robustness_score: float
    notes: tuple[str, ...]
    result: GridOptimizationResult


@dataclass(frozen=True)
class BrainOutcome:
    """Feedback from executing a BrainDecision."""

    success: bool
    cost_paid: float
    risk_realized: float
    asset_gained: float
    rule_violations: int = 0
    failure_mode: str = "unknown"

    @property
    def utility(self) -> float:
        return self.asset_gained - self.cost_paid - self.risk_realized - self.rule_violations


@dataclass(frozen=True)
class GossipNote:
    """A second-order signal about a tool's health, originating from another agent.

    Gossip is treated as a priced tool, not as ground truth.  The Brain computes
    an effective weight from source reputation, signal age, and whether the source
    is a Predatory Scout before applying any update.

    Attributes:
        tool: Name of the tool the claim is about.
        claim: ``"failing"`` or ``"healthy"``.
        source_id: Unique identifier of the reporting agent.
        source_reputation: Normalised reputation score of the reporting agent [0, 1].
        source_is_scout: ``True`` when the source has been auto-flagged as a
            Predatory Scout (high gossip rate, low direct usage, oscillating rep).
        confidence: The reporting agent's own stated confidence in the claim [0, 1].
        age_minutes: How many minutes old the gossip is at ingestion time.
    """

    tool: str
    claim: str
    source_id: str
    source_reputation: float = 0.5
    source_is_scout: bool = False
    confidence: float = 1.0
    age_minutes: float = 0.0


@dataclass
class ScoutRecord:
    """Per-source accuracy ledger used to set the scout discount dynamically.

    A Predatory Scout starts with a discount of 0.7 (its gossip is heard but
    down-weighted).  Once it has logged at least 50 claims and its precision
    exceeds 0.80, the discount lifts to 0.9 — it has earned increased trust.
    """

    DISCOUNT_DEFAULT: float = 0.7
    DISCOUNT_PROMOTED: float = 0.9
    PROMOTION_MIN_CLAIMS: int = 50
    PROMOTION_PRECISION: float = 0.80

    _predictions: list[bool] = field(default_factory=list)  # True == gossip matched reality

    def log_prediction(self, matched_reality: bool) -> None:
        self._predictions.append(matched_reality)

    @property
    def precision(self) -> float:
        if not self._predictions:
            return 0.0
        return sum(self._predictions) / len(self._predictions)

    @property
    def discount(self) -> float:
        if (
            len(self._predictions) >= self.PROMOTION_MIN_CLAIMS
            and self.precision >= self.PROMOTION_PRECISION
        ):
            return self.DISCOUNT_PROMOTED
        return self.DISCOUNT_DEFAULT


@dataclass
class BrainMemory:
    """Cross-world memory for domains, actions, and tools."""

    domain_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    action_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    tool_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    scout_tracker: dict[str, ScoutRecord] = field(default_factory=dict)
    base_learning_rate: float = 0.15

    # Gossip hyperparameters
    GOSSIP_LR: float = field(default=0.06, init=False, repr=False)
    GOSSIP_DECAY_RATE: float = field(default=0.97, init=False, repr=False)

    def prior_risk_adjustment(self, task: BrainTask) -> float:
        stats = self.domain_stats.get(task.domain)
        if not stats:
            return 0.0
        failure_rate = 1.0 - stats.get("success_rate", 1.0)
        return max(-0.2, min(0.25, failure_rate - 0.15))

    def tool_reliability_adjustment(self, tool: ToolProfile) -> float:
        stats = self.tool_stats.get(tool.name)
        if not stats:
            return 0.0
        # Tool failures need to leave a meaningful scar; otherwise high-prior
        # tools remain attractive even after repeated bad outcomes.
        return max(-0.75, min(0.15, stats.get("success_rate", tool.reliability) - tool.reliability))

    def update(
        self,
        task: BrainTask,
        decision: BrainDecision,
        outcome: BrainOutcome,
    ) -> None:
        self._update_bucket(self.domain_stats, task.domain, outcome, self.base_learning_rate)
        self._update_bucket(self.action_stats, decision.action, outcome, self.base_learning_rate)
        if decision.selected_tool:
            self._update_bucket(
                self.tool_stats,
                decision.selected_tool,
                outcome,
                self._tool_learning_rate(decision.selected_tool, outcome),
            )

    def _update_bucket(
        self, store: dict[str, dict[str, float]], key: str, outcome: BrainOutcome, learning_rate: float
    ) -> None:
        current = store.get(
            key,
            {
                "count": 0.0,
                "success_rate": 1.0,
                "utility": 0.0,
                "consecutive_failures": 0.0,
            },
        )
        lr = clamp01(learning_rate)
        outcome_success = 1.0 if outcome.success else 0.0
        current["success_rate"] = current["success_rate"] * (1.0 - lr) + outcome_success * lr
        current["utility"] = current["utility"] * (1.0 - lr) + outcome.utility * lr
        current["consecutive_failures"] = (
            0.0 if outcome.success else current.get("consecutive_failures", 0.0) + 1.0
        )
        current["count"] = current.get("count", 0.0) + 1.0
        store[key] = current

    def decay(self, rate: float = 0.03) -> None:
        """Forgive old scars gradually in non-stationary environments."""

        for store in (self.domain_stats, self.action_stats, self.tool_stats):
            for stats in store.values():
                stats["success_rate"] = stats.get("success_rate", 1.0) * (1.0 - rate) + rate
                stats["utility"] = stats.get("utility", 0.0) * (1.0 - rate)
                stats["consecutive_failures"] = max(
                    0.0, stats.get("consecutive_failures", 0.0) - rate
                )

    def _tool_learning_rate(self, tool_name: str, outcome: BrainOutcome) -> float:
        if outcome.success:
            return 0.10
        mode_rates = {
            "tool_error": 0.20,
            "hallucination": 0.20,
            "bad_data": 0.18,
            "timeout": 0.05,
            "rate_limit": 0.05,
            "user_abandon": 0.00,
            "environment_noise": 0.03,
            "unknown": self.base_learning_rate,
        }
        rate = mode_rates.get(outcome.failure_mode, self.base_learning_rate)
        failures = self.tool_stats.get(tool_name, {}).get("consecutive_failures", 0.0)
        if failures >= 3:
            rate *= 1.5
        return min(0.6, rate)

    # ------------------------------------------------------------------
    # Gossip ingestion
    # ------------------------------------------------------------------

    def _compute_gossip_weight(self, note: GossipNote) -> float:
        """Return the effective influence weight for *note*.

        w = clamp(source_reputation) × GOSSIP_DECAY_RATE^age_minutes × scout_discount
        """
        w = clamp01(note.source_reputation) * (self.GOSSIP_DECAY_RATE ** note.age_minutes)
        if note.source_is_scout:
            record = self.scout_tracker.setdefault(note.source_id, ScoutRecord())
            w *= record.discount
        return w

    def ingest_gossip(self, note: GossipNote, actual_outcome: bool | None = None) -> None:
        """Apply a weighted virtual update to a tool's memory from a gossip note.

        The update follows:

            effective_lr = GOSSIP_LR * w
            S_new = S_old * (1 - effective_lr) + virtual_success * effective_lr

        where:

            w = source_reputation * decay(age_minutes) * scout_discount
            virtual_success = 0.0 if claim == "failing" else 1.0

        A single malicious note barely moves the needle (typical w ≈ 0.44 →
        effective_lr ≈ 2.6%).  Three independent non-scout notes within the
        decay window sum w > 1.2, triggering a meaningful update — that is
        consensus, not manipulation.

        If *actual_outcome* is supplied, the scout's prediction accuracy is
        logged so its discount can be adjusted over time.
        """
        w = self._compute_gossip_weight(note)

        virtual_success = 0.0 if note.claim == "failing" else 1.0
        self.update_tool_memory(
            tool=note.tool,
            virtual_success=virtual_success,
            weight=w,
        )

        if note.source_is_scout and actual_outcome is not None:
            record = self.scout_tracker.setdefault(note.source_id, ScoutRecord())
            claim_was_correct = (note.claim == "failing" and not actual_outcome) or (
                note.claim != "failing" and actual_outcome
            )
            record.log_prediction(claim_was_correct)

    def update_tool_memory(
        self,
        tool: str,
        virtual_success: float,
        weight: float = 1.0,
    ) -> None:
        """Apply a weighted success/failure signal to a tool's stats bucket.

        The effective learning rate is ``GOSSIP_LR * weight``, capped at
        ``GOSSIP_LR`` so a single gossip note never exceeds the gossip budget.
        For consensus signals (weight > 1.0 from multiple independent sources)
        the effective rate is allowed to scale linearly up to 3× GOSSIP_LR.

        Parameters:
            tool: The tool name to update.
            virtual_success: 1.0 for a success signal, 0.0 for failure.
            weight: Dimensionless influence weight.
        """
        effective_lr = min(3 * self.GOSSIP_LR, self.GOSSIP_LR * max(0.0, weight))
        stats = self.tool_stats.get(
            tool,
            {
                "count": 0.0,
                "success_rate": 1.0,
                "utility": 0.0,
                "consecutive_failures": 0.0,
            },
        )
        stats["success_rate"] = stats["success_rate"] * (1.0 - effective_lr) + virtual_success * effective_lr
        if virtual_success == 0.0:
            stats["consecutive_failures"] = stats.get("consecutive_failures", 0.0) + effective_lr
        else:
            stats["consecutive_failures"] = max(
                0.0, stats.get("consecutive_failures", 0.0) - effective_lr
            )
        self.tool_stats[tool] = stats


# ---------------------------------------------------------------------------
# Phase 2: Price learning — inverse RL from observed outcomes
# ---------------------------------------------------------------------------

_FAILURE_BLAME_TABLE: dict[str, float] = {
    "tool_error": 0.95,
    "hallucination": 0.90,
    "bad_data": 0.85,
    "unknown": 0.50,
    "user_abandon": 0.10,
    "timeout": 0.15,
    "rate_limit": 0.10,
    "environment_noise": 0.05,
}


def attribute_to_tool(failure_mode: str) -> float:
    """Return the probability [0, 1] that a failure was caused by the tool itself.

    Used by ``PriceAdapter`` to weight observed cost/risk signals during
    learning.  Environment-driven modes (timeout, rate_limit,
    environment_noise) contribute little to the tool's learned price, while
    tool-intrinsic modes (tool_error, hallucination) are attributed almost
    entirely to the tool.

    Parameters
    ----------
    failure_mode:
        The ``BrainOutcome.failure_mode`` string from the outcome feedback.

    Returns
    -------
    float
        A blame weight in [0, 1].  Unrecognised modes return 0.50.
    """
    return _FAILURE_BLAME_TABLE.get(failure_mode, 0.50)


@dataclass
class LearnedPrices:
    """Learned additive corrections to a tool's declared C/R/A prices.

    Each delta is applied on top of the corresponding ``ToolProfile`` value:

    * A positive ``cost_delta`` means the tool consistently costs more than
      declared — future utility calculations will discount it accordingly.
    * A positive ``risk_delta`` means the tool's realized risk exceeds its
      stated ``risk`` field.
    * A negative ``asset_delta`` means the tool delivers less value than
      expected on success.
    """

    cost_delta: float = 0.0
    risk_delta: float = 0.0
    asset_delta: float = 0.0
    n_observations: int = 0


@dataclass
class PriceAdapter:
    """Inverse-RL price learner that infers true C/R/A from observed outcomes.

    ``PriceAdapter`` sits alongside a ``ManifoldBrain`` and continuously
    reconciles each tool's declared prices (cost, risk, asset) with what is
    actually observed in ``BrainOutcome`` feedback.  Corrections are learned
    via exponential moving averages (EMA), with causal attribution controlling
    how much of each failure is charged to the tool vs the environment.

    The three learning channels:

    * **Cost** — always updated; we always pay cost regardless of success.
    * **Risk** — weighted by the ``attribute_to_tool`` blame score so that
      environment-driven failures (timeout, rate_limit) do not incorrectly
      inflate a tool's learned risk.
    * **Asset** — only updated on success; failure-path asset loss is already
      captured by ``BrainMemory`` reliability tracking.

    After ``min_observations`` have been accumulated, ``adapt()`` returns a
    corrected ``ToolProfile`` whose prices reflect lived experience.  Before
    that threshold, ``adapt()`` returns the original profile unchanged to
    avoid over-reacting to sparse noise.

    Parameters
    ----------
    lr:
        EMA learning rate for price corrections.  Defaults to 0.12.
    min_observations:
        Minimum observations before ``adapt()`` applies corrections.
        Defaults to 3.

    Example
    -------
    ::

        adapter = PriceAdapter()
        brain = ManifoldBrain(config, tools=tools, price_adapter=adapter)

        # After each decision/outcome cycle, the adapter silently updates.
        decision = brain.decide(task)
        brain.learn(task, decision, outcome)   # calls adapter.observe internally

        # Inspect what was learned:
        corrections = adapter.price_corrections()
        _logger.debug("web_search cost_delta: %s", corrections["web_search"].cost_delta)
    """

    lr: float = 0.12
    min_observations: int = 3

    _corrections: dict[str, LearnedPrices] = field(
        default_factory=dict, init=False, repr=False
    )

    def observe(self, tool: ToolProfile, outcome: BrainOutcome) -> None:
        """Update learned price corrections from an observed outcome.

        Parameters
        ----------
        tool:
            The ``ToolProfile`` that was used to produce *outcome*.
        outcome:
            The observed feedback from executing the tool.
        """
        corr = self._corrections.setdefault(tool.name, LearnedPrices())
        blame = 1.0 if outcome.success else attribute_to_tool(outcome.failure_mode)

        # Cost: compare observed cost_paid to declared cost + latency.
        stated_cost = tool.cost + tool.latency
        cost_gap = outcome.cost_paid - stated_cost
        corr.cost_delta = corr.cost_delta * (1.0 - self.lr) + cost_gap * self.lr

        # Risk: weight by causal blame so env noise doesn't taint tool risk.
        risk_gap = outcome.risk_realized - tool.risk
        corr.risk_delta = corr.risk_delta * (1.0 - self.lr) + risk_gap * self.lr * blame

        # Asset: success path only — failure asset is handled by reliability.
        if outcome.success:
            asset_gap = outcome.asset_gained - tool.asset
            corr.asset_delta = corr.asset_delta * (1.0 - self.lr) + asset_gap * self.lr

        corr.n_observations += 1

    def adapt(self, tool: ToolProfile) -> ToolProfile:
        """Return a ``ToolProfile`` with learned price corrections applied.

        Returns the original *tool* unchanged if fewer than
        ``min_observations`` have been accumulated.

        Parameters
        ----------
        tool:
            The tool whose prices should be adjusted.

        Returns
        -------
        ToolProfile
            A new profile with corrected cost, risk, and asset, or the
            original profile if there is insufficient data.
        """
        corr = self._corrections.get(tool.name)
        if corr is None or corr.n_observations < self.min_observations:
            return tool
        return ToolProfile(
            name=tool.name,
            cost=clamp01(tool.cost + corr.cost_delta),
            latency=tool.latency,
            reliability=tool.reliability,
            risk=clamp01(tool.risk + corr.risk_delta),
            asset=clamp01(tool.asset + corr.asset_delta),
            domain=tool.domain,
        )

    def price_corrections(self) -> dict[str, LearnedPrices]:
        """Return a copy of all learned price corrections keyed by tool name."""
        return dict(self._corrections)


@dataclass
class ManifoldBrain:
    """Adaptive meta-controller for agent decisions.

    Supports both single-agent (legacy) mode and **Fleet Orchestrator** mode
    where multiple agents are tracked via :attr:`fleet` / :attr:`fleet_watchdogs`.
    The default ``movement`` and ``watchdog`` attributes serve as the "primary"
    agent (backward-compatible).  Use :meth:`register_agent` to add fleet members.
    """

    config: BrainConfig = field(default_factory=BrainConfig)
    tools: list[ToolProfile] = field(default_factory=list)
    memory: BrainMemory = field(default_factory=BrainMemory)
    price_adapter: PriceAdapter | None = None
    asset_adapter: AssetAdapter | None = None
    movement: MovementStateMachine = field(default_factory=MovementStateMachine)
    movement_planner: PathPlanner | None = field(default=None, repr=False, compare=False)
    movement_bus: CellUpdateBus | None = field(default=None, repr=False, compare=False)
    watchdog: Watchdog = field(default_factory=Watchdog)
    mqtt_gateway: object | None = field(default=None, repr=False, compare=False)

    # Fleet Orchestrator state — populated via register_agent()
    fleet: dict[str, MovementStateMachine] = field(default_factory=dict, repr=False)
    fleet_watchdogs: dict[str, Watchdog] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.movement_planner is None:
            from .planner import CRNAPlanner

            self.movement_planner = CRNAPlanner()
        if self.movement_bus is None:
            self.movement_bus = get_bus()
        subscriber_id = f"manifold-brain-movement-{uuid.uuid4().hex}"
        brain_ref = weakref.ref(self)

        def _movement_callback(update: CellUpdate) -> None:
            brain = brain_ref()
            if brain is not None:
                brain._handle_movement_update(update)

        self.movement_bus.subscribe(subscriber_id, _movement_callback)
        if hasattr(self.movement_bus, "unsubscribe"):
            weakref.finalize(self, self.movement_bus.unsubscribe, subscriber_id)

    # ------------------------------------------------------------------
    # Fleet Orchestrator — N-Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str) -> None:
        """Register a new agent into the fleet.

        Creates a fresh :class:`MovementStateMachine` and :class:`Watchdog` for
        the given *agent_id*.  If the agent is already registered, this is a no-op.

        Parameters
        ----------
        agent_id:
            Unique identifier for the agent (e.g. ``"robot-1"``).
        """
        if agent_id in self.fleet:
            return
        self.fleet[agent_id] = MovementStateMachine()
        self.fleet_watchdogs[agent_id] = Watchdog(
            timeout_seconds=self.watchdog.timeout_seconds,
        )
        _logger.info("Fleet agent registered: %s (total=%d)", agent_id, len(self.fleet))

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the fleet.

        Parameters
        ----------
        agent_id:
            Identifier of the agent to remove.
        """
        self.fleet.pop(agent_id, None)
        self.fleet_watchdogs.pop(agent_id, None)
        _logger.info("Fleet agent unregistered: %s (total=%d)", agent_id, len(self.fleet))

    def fleet_agent_ids(self) -> list[str]:
        """Return a list of all currently registered fleet agent IDs."""
        return list(self.fleet.keys())

    @property
    def logical_pos(self) -> tuple[int, int, int]:
        """Current snapped grid position for movement."""
        return self.movement.logical_pos

    @logical_pos.setter
    def logical_pos(self, value: tuple[int, int, int]) -> None:
        self.movement.logical_pos = tuple(int(coord) for coord in value)

    @property
    def physical_pos(self) -> tuple[float, float, float]:
        """Current continuous position for movement."""
        return self.movement.physical_pos

    @physical_pos.setter
    def physical_pos(self, value: tuple[float, float, float]) -> None:
        self.movement.physical_pos = tuple(float(coord) for coord in value)

    @property
    def next_safe_cell(self) -> tuple[int, int, int] | None:
        """The next path cell the agent is moving toward."""
        return self.movement.next_safe_cell

    @next_safe_cell.setter
    def next_safe_cell(self, value: tuple[int, int, int] | None) -> None:
        self.movement.next_safe_cell = tuple(int(coord) for coord in value) if value is not None else None

    @property
    def movement_state(self) -> MovementState:
        """Current movement state machine phase."""
        return self.movement.state

    @property
    def replan_pending(self) -> bool:
        """Whether a replan has been requested by a gatekeeper event."""
        return self.movement.replan_pending

    @replan_pending.setter
    def replan_pending(self, value: bool) -> None:
        self.movement.replan_pending = bool(value)

    @property
    def current_path_set(self) -> set[tuple[int, int, int]]:
        """Current set of path cells used for O(1) membership checks."""
        return self.movement.current_path_set

    def set_movement_goal(self, target_cell: tuple[int, int, int], agent_id: str | None = None) -> None:
        """Assign a new navigation target and request a fresh plan.

        Parameters
        ----------
        target_cell:
            The 3-D grid coordinate to navigate toward.
        agent_id:
            If provided, route the goal to the specified fleet agent.
            If ``None``, route to the primary (legacy) movement machine.
        """
        if agent_id is not None and agent_id in self.fleet:
            self.fleet[agent_id].set_goal(target_cell)
        else:
            self.movement.set_goal(target_cell)

    def feed_watchdog(self, agent_id: str | None = None) -> None:
        """Reset the watchdog timer — call this from the hardware heartbeat loop.

        Parameters
        ----------
        agent_id:
            If provided, feed the watchdog of a specific fleet agent.
            If ``None``, feed the primary watchdog.
        """
        if agent_id is not None and agent_id in self.fleet_watchdogs:
            self.fleet_watchdogs[agent_id].feed()
        else:
            self.watchdog.feed()

    def tick(self, delta_time: float) -> None:
        """Advance the control loop: safety → movement → telemetry sync.

        The loop order is strictly:
        1. **Safety check** — if the watchdog has expired, enter ERROR state and
           issue EMERGENCY_STOP via the MQTT gateway.
        2. **Movement logic** — delegate to the MovementStateMachine (which
           handles replan_pending internally).
        3. **MQTT telemetry sync** — push position/status to hardware.
        4. **Fleet iteration** — repeat steps 1–2 for each fleet member.

        Automatic recovery: once :meth:`feed_watchdog` is called again, the
        ERROR state is cleared and normal operation resumes on the next tick.
        """
        # --- Primary agent ---
        self._tick_single(self.movement, self.watchdog, delta_time)

        # --- Fleet agents ---
        for agent_id, movement in self.fleet.items():
            wd = self.fleet_watchdogs.get(agent_id)
            if wd is None:
                continue
            self._tick_single(movement, wd, delta_time, agent_id=agent_id)

        # --- MQTT telemetry sync (skip if primary is in ERROR) ---
        if (
            self.movement.state is not MovementState.ERROR
            and self.mqtt_gateway is not None
            and hasattr(self.mqtt_gateway, "sync_to_hardware")
        ):
            try:
                self.mqtt_gateway.sync_to_hardware()
            except Exception as exc:  # noqa: BLE001
                _logger.debug("MQTT sync failed: %s", exc)

    def _tick_single(
        self,
        movement: MovementStateMachine,
        watchdog: Watchdog,
        delta_time: float,
        agent_id: str | None = None,
    ) -> None:
        """Run one tick cycle for a single agent (primary or fleet member).

        Parameters
        ----------
        movement:
            The MovementStateMachine for this agent.
        watchdog:
            The Watchdog for this agent.
        delta_time:
            Elapsed time since last tick.
        agent_id:
            Agent identifier (for logging), or ``None`` for the primary.
        """
        # Safety: watchdog check
        if watchdog.is_expired():
            if movement.state is not MovementState.ERROR:
                label = agent_id or "primary"
                _logger.warning(
                    "Watchdog expired (%.3fs) for %s — entering ERROR state",
                    watchdog.elapsed, label,
                )
                movement.state = MovementState.ERROR
                self._emit_emergency_stop()
            return  # Do NOT advance movement while in ERROR

        # Recovery from ERROR: watchdog was fed externally
        if movement.state is MovementState.ERROR:
            label = agent_id or "primary"
            _logger.info("Watchdog recovered for %s — resuming from ERROR → IDLE", label)
            movement.state = MovementState.IDLE

        # Movement logic (includes interrupt/replan processing)
        movement.tick(delta_time, planner=self.movement_planner)

    def _emit_emergency_stop(self) -> None:
        """Publish EMERGENCY_STOP via the MQTT gateway if available."""
        if self.mqtt_gateway is None:
            return
        bridge = getattr(self.mqtt_gateway, "bridge", None)
        if bridge is not None and hasattr(bridge, "publish_command"):
            try:
                bridge.publish_command("EMERGENCY_STOP")
            except Exception as exc:  # noqa: BLE001
                _logger.debug("EMERGENCY_STOP publish failed: %s", exc)

    def _handle_movement_update(self, update: CellUpdate) -> None:
        """Bridge bus updates into the movement gatekeeper logic.

        Propagates cell updates to the primary movement machine and all fleet
        members so that every agent benefits from shared obstacle awareness.
        """
        self.movement.on_cell_update(update)
        for movement in self.fleet.values():
            movement.on_cell_update(update)

    # ------------------------------------------------------------------
    # Policy Command Interface
    # ------------------------------------------------------------------

    def handle_command(
        self,
        action_code: int,
        params: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Central command dispatcher: maps a PolicyAction code to a Brain method.

        This is the API endpoint for the UI / MQTT bridge.  The frontend sends
        ``{"action_code": N, "params": {...}}`` on the ``cmd`` topic; the bridge
        deserializes it and calls this method.

        Parameters
        ----------
        action_code:
            Integer matching a :class:`PolicyAction` member.
        params:
            Action-specific parameters (target coords, zone id, etc.).
        agent_id:
            If provided, route the command only to that fleet agent.
            If ``None`` or ``"ALL"``, broadcast the command to the primary
            agent and all fleet members.
        """
        if params is None:
            params = {}
        try:
            action = PolicyAction(action_code)
        except ValueError:
            _logger.warning("Unknown policy action code: %d", action_code)
            return

        handler = self._policy_action_map().get(action)
        if handler is None:
            _logger.info("PolicyAction %s not yet implemented", action.name)
            return

        # Routing: broadcast or targeted
        if agent_id is not None and agent_id != "ALL" and agent_id in self.fleet:
            # Targeted to a specific fleet agent
            handler(params, agent_id=agent_id)
        else:
            # Broadcast: primary + all fleet members
            handler(params, agent_id=None)
            if agent_id == "ALL":
                for fleet_id in list(self.fleet.keys()):
                    handler(params, agent_id=fleet_id)

    def _policy_action_map(self) -> dict[PolicyAction, Any]:
        """Return the mapping of PolicyAction → handler method."""
        return {
            PolicyAction.DEPLOY_AGENT: self._cmd_deploy,
            PolicyAction.GATHER_DATA: self._cmd_gather,
            PolicyAction.RECALIBRATE: self._cmd_recalibrate,
            PolicyAction.PATROL: self._cmd_patrol,
            PolicyAction.MAINTENANCE: self._cmd_maintenance,
            PolicyAction.SCAN_WORLD: self._cmd_scan_world,
            PolicyAction.INTERACT_OBJECT: self._cmd_interact_object,
            PolicyAction.DEFEND_ZONE: self._cmd_defend_zone,
            PolicyAction.ESCALATE: self._cmd_escalate,
            PolicyAction.RETURN_HOME: self._cmd_return_home,
            PolicyAction.FORM_COALITION: self._cmd_form_coalition,
            PolicyAction.RESEARCH: self._cmd_research,
            PolicyAction.EMERGENCY_STOP: self._cmd_emergency_stop,
        }

    def _cmd_deploy(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Deploy the agent to a specified position."""
        target = params.get("target")
        if target and len(target) >= 3:
            self.set_movement_goal(tuple(int(c) for c in target[:3]), agent_id=agent_id)
            _logger.info("DEPLOY_AGENT → target=%s agent=%s", target, agent_id or "primary")

    def _cmd_gather(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Navigate to a data-gathering waypoint."""
        target = params.get("target")
        if target and len(target) >= 3:
            self.set_movement_goal(tuple(int(c) for c in target[:3]), agent_id=agent_id)
            _logger.info("GATHER_DATA → target=%s agent=%s", target, agent_id or "primary")

    def _cmd_recalibrate(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Trigger sensor recalibration (future: sensor_bridge integration)."""
        _logger.info("RECALIBRATE requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_patrol(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Set a patrol goal — translate UI coordinates to brain pathing."""
        end = params.get("end") or params.get("target")
        if end and len(end) >= 3:
            self.set_movement_goal(tuple(int(c) for c in end[:3]), agent_id=agent_id)
            _logger.info("PATROL → end=%s agent=%s", end, agent_id or "primary")

    def _cmd_maintenance(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Enter a maintenance/docking routine."""
        _logger.info("MAINTENANCE requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_scan_world(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Initiate a full-world scan sweep."""
        _logger.info("SCAN_WORLD requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_interact_object(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Interact with a specific object (future: manipulation subsystem)."""
        _logger.info("INTERACT_OBJECT requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_defend_zone(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Move to and hold a defensive zone."""
        target = params.get("zone_center") or params.get("target")
        if target and len(target) >= 3:
            self.set_movement_goal(tuple(int(c) for c in target[:3]), agent_id=agent_id)
            _logger.info("DEFEND_ZONE → center=%s agent=%s", target, agent_id or "primary")

    def _cmd_escalate(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Escalate current situation to a higher-level controller."""
        _logger.info("ESCALATE requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_return_home(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Navigate back to the home/dock position."""
        home = params.get("home", (0, 0, 0))
        self.set_movement_goal(tuple(int(c) for c in home[:3]), agent_id=agent_id)
        _logger.info("RETURN_HOME → %s agent=%s", home, agent_id or "primary")

    def _cmd_form_coalition(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Request coalition formation with specified agent ids."""
        _logger.info("FORM_COALITION requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_research(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Queue a research/exploration task."""
        _logger.info("RESEARCH requested (params=%s, agent=%s)", params, agent_id or "primary")

    def _cmd_emergency_stop(self, params: dict[str, Any], agent_id: str | None = None) -> None:
        """Trigger immediate EMERGENCY_STOP — same as watchdog timeout."""
        _logger.warning("EMERGENCY_STOP command received (agent=%s)", agent_id or "primary")
        if agent_id is not None and agent_id in self.fleet:
            self.fleet[agent_id].state = MovementState.ERROR
        else:
            self.movement.state = MovementState.ERROR
        self._emit_emergency_stop()

    def decide(self, task: BrainTask) -> BrainDecision:
        task = task.normalized()
        world = self.map_task_to_world(task)
        result = AgentPopulation(
            seed=str(self.config.seed),
            n=self.config.population_size,
            predators=self.config.predators,
        ).optimize(
            world,
            generations=self.config.generations,
            verification_cost=self._verification_cost(task),
            false_trust_penalty=self._false_trust_penalty(task),
        )
        risk_score = clamp01(self._risk_score(task) + self.memory.prior_risk_adjustment(task))
        selected_tool = self.select_tool(task)
        action = self.choose_action(task, result, risk_score, selected_tool)
        active_tool = selected_tool if action == "use_tool" else None
        expected_utility = self.expected_action_utility(task, action, active_tool, risk_score)
        return BrainDecision(
            action=action,
            confidence=clamp01(1.0 - risk_score + result.audit.robustness_score * 0.25),
            risk_score=risk_score,
            expected_utility=expected_utility,
            selected_tool=active_tool.name if active_tool else None,
            verification_rate=result.verification,
            reputation_cap=result.reputation_cap,
            robustness_score=result.audit.robustness_score,
            notes=self._notes(task, action, active_tool, risk_score),
            result=result,
        )

    def learn(self, task: BrainTask, decision: BrainDecision, outcome: BrainOutcome) -> None:
        self.memory.update(task.normalized(), decision, outcome)
        if self.price_adapter and decision.selected_tool:
            tool = next((t for t in self.tools if t.name == decision.selected_tool), None)
            if tool:
                self.price_adapter.observe(tool, outcome)

    def map_task_to_world(self, task: BrainTask) -> GridWorld:
        size = self.config.grid_size
        center = size // 2
        world = GridWorld(size=size, seed=self.config.seed)
        for row in range(size):
            for col in range(size):
                distance = (abs(row - center) + abs(col - center)) / max(1, size)
                tool_value = self.best_tool_value(task)
                cost = 0.03 + 0.12 * task.complexity + 0.08 * task.time_pressure + 0.05 * distance
                risk = clamp01(
                    0.08
                    + 0.35 * task.uncertainty
                    + 0.25 * (1.0 - task.source_confidence)
                    + 0.30 * task.safety_sensitivity
                    + 0.15 * distance
                )
                neutrality = clamp01(0.9 - 0.4 * task.complexity - 0.2 * distance)
                asset = 0.05 + task.stakes * (0.3 + tool_value * 0.4) * (1.0 - distance)
                world.set_cell(row, col, cost=cost, risk=risk, neutrality=neutrality, asset=asset)
        world.add_dynamic_targets(
            [
                {
                    "id": "resolved_goal",
                    "pos": (
                        center,
                        max(0, min(size - 1, round(center + (task.uncertainty - 0.5) * center))),
                    ),
                    "asset": 2.0 + task.stakes * 7.0 + self.best_tool_value(task),
                    "moves": "random_walk" if task.dynamic_goal else "static",
                }
            ]
        )
        world.add_rule("failed_goal", 1.0 + task.stakes * 5.0, "miss_target")
        world.add_rule("bad_tool_or_source", 1.0 + task.stakes * 4.0, "trusted_lie")
        world.add_rule("resource_exhaustion", 1.0, "low_energy")
        return world

    def select_tool(self, task: BrainTask) -> ToolProfile | None:
        candidates = [tool for tool in self.tools if tool.domain in {task.domain, "general"}]
        if not candidates or task.tool_relevance < 0.35:
            return None
        adjusted = []
        for tool in candidates:
            # Use price-adapted profile for utility pricing if adapter has enough data.
            effective = self.price_adapter.adapt(tool) if self.price_adapter else tool
            reliability = clamp01(effective.reliability + self.memory.tool_reliability_adjustment(effective))
            utility = effective.asset * reliability - effective.cost - effective.latency - effective.risk
            adjusted.append((utility, tool))  # return original tool to preserve name/domain
        best_utility, best_tool = max(adjusted, key=lambda item: item[0])
        return best_tool if best_utility > 0.0 else None

    def choose_action(
        self,
        task: BrainTask,
        result: GridOptimizationResult,
        risk_score: float,
        tool: ToolProfile | None,
    ) -> BrainAction:
        if (
            risk_score <= 0.35
            and task.source_confidence >= 0.75
            and task.safety_sensitivity <= 0.25
            and task.time_pressure >= 0.60
        ):
            return "answer"
        if task.safety_sensitivity >= 0.9 and risk_score >= 0.78:
            return "refuse"
        if task.safety_sensitivity >= 0.70 and risk_score >= self._escalation_threshold(task):
            return "escalate"
        if task.complexity >= 0.75 and task.uncertainty >= 0.50 and task.time_pressure <= 0.65:
            return "plan"
        if tool and task.tool_relevance >= 0.60 and task.safety_sensitivity < 0.80:
            return "use_tool"
        if task.collaboration_value >= 0.75 and task.complexity >= 0.60 and risk_score >= 0.45:
            return "delegate"
        if task.time_pressure <= 0.20 and task.uncertainty >= 0.50 and task.stakes <= 0.55:
            return "explore"
        if risk_score >= self._escalation_threshold(task):
            return "escalate"
        if task.uncertainty >= 0.68 and task.user_patience >= 0.35:
            return "clarify"
        if (1.0 - task.source_confidence) >= 0.55:
            return "retrieve"
        if risk_score >= result.audit.verification_threshold:
            return "verify"
        return "answer"

    def expected_action_utility(
        self,
        task: BrainTask,
        action: BrainAction,
        tool: ToolProfile | None,
        risk_score: float,
    ) -> float:
        cost = brain_action_cost(action, task)
        tool_bonus = tool.utility if action == "use_tool" and tool else 0.0
        stated_asset = task.stakes + tool_bonus
        # Use learned asset correction when the adapter has enough data.
        action_asset = (
            self.asset_adapter.adapt_asset(action, stated_asset)
            if self.asset_adapter
            else stated_asset
        )
        risk_discount = risk_score * brain_action_risk_multiplier(action)
        return action_asset - cost - risk_discount

    def observe_asset(
        self, action: BrainAction, signal: str, stated_asset: float
    ) -> None:
        """Forward a user-preference signal to the ``AssetAdapter`` if present.

        Parameters
        ----------
        action:
            The ``BrainAction`` string that was executed.
        signal:
            One of ``"correction"``, ``"acceptance"``, ``"silence"``, or
            ``"ambiguous"``.
        stated_asset:
            The task's ``stakes`` value used at decision time.
        """
        if self.asset_adapter:
            self.asset_adapter.observe_outcome(action, signal, stated_asset)

    def best_tool_value(self, task: BrainTask) -> float:
        tool = self.select_tool(task)
        return max(0.0, tool.utility if tool else 0.0)

    def _risk_score(self, task: BrainTask) -> float:
        return clamp01(
            0.30 * task.uncertainty
            + 0.20 * task.complexity
            + 0.25 * task.stakes
            + 0.20 * (1.0 - task.source_confidence)
            + 0.30 * task.safety_sensitivity
            + 0.10 * task.tool_relevance
            + (0.10 if task.dynamic_goal else 0.0)
        )

    def _verification_cost(self, task: BrainTask) -> float:
        return max(0.02, 0.18 * (1.1 - task.user_patience) + 0.05 * task.time_pressure)

    def _false_trust_penalty(self, task: BrainTask) -> float:
        return 0.5 + 3.0 * task.stakes + 2.0 * task.safety_sensitivity

    def _escalation_threshold(self, task: BrainTask) -> float:
        return clamp01(0.82 - 0.25 * task.safety_sensitivity - 0.12 * task.stakes)

    def _notes(
        self,
        task: BrainTask,
        action: BrainAction,
        tool: ToolProfile | None,
        risk_score: float,
    ) -> tuple[str, ...]:
        notes = [f"Brain mapped '{task.domain}' task with risk={risk_score:.2f}."]
        if tool:
            notes.append(f"Selected tool candidate: {tool.name}.")
        if action in {"escalate", "refuse"}:
            notes.append("Safety/stakes pressure dominated the policy.")
        if action == "use_tool":
            notes.append("Tool utility exceeded direct-answer utility after risk pricing.")
        return tuple(notes)


def brain_action_cost(action: BrainAction, task: BrainTask) -> float:
    patience_cost = 1.0 - task.user_patience
    costs = {
        "answer": 0.05 + 0.04 * task.complexity,
        "clarify": 0.14 + 0.16 * patience_cost,
        "retrieve": 0.18 + 0.05 * task.complexity,
        "verify": 0.17 + 0.06 * task.complexity,
        "use_tool": 0.20 + 0.08 * task.time_pressure,
        "delegate": 0.32 + 0.12 * patience_cost,
        "plan": 0.16 + 0.12 * task.complexity,
        "explore": 0.16 + 0.08 * task.uncertainty,
        "exploit": 0.08 + 0.06 * task.time_pressure,
        "wait": 0.08 + 0.20 * task.time_pressure,
        "escalate": 0.42 + 0.12 * patience_cost,
        "refuse": 0.20 + 0.15 * task.stakes,
        "stop": 0.04,
    }
    return costs[action]


def brain_action_risk_multiplier(action: BrainAction) -> float:
    multipliers = {
        "answer": 1.0,
        "clarify": 0.45,
        "retrieve": 0.35,
        "verify": 0.25,
        "use_tool": 0.40,
        "delegate": 0.30,
        "plan": 0.55,
        "explore": 0.70,
        "exploit": 0.85,
        "wait": 0.65,
        "escalate": 0.15,
        "refuse": 0.10,
        "stop": 0.20,
    }
    return multipliers[action]


def default_tools() -> list[ToolProfile]:
    return [
        ToolProfile("web_search", cost=0.12, latency=0.18, reliability=0.78, risk=0.12, asset=0.75, domain="general"),
        ToolProfile("calculator", cost=0.04, latency=0.04, reliability=0.96, risk=0.03, asset=0.50, domain="math"),
        ToolProfile("code_runner", cost=0.15, latency=0.20, reliability=0.86, risk=0.08, asset=0.85, domain="coding"),
        ToolProfile("retriever", cost=0.10, latency=0.12, reliability=0.82, risk=0.10, asset=0.70, domain="research"),
        ToolProfile("human_reviewer", cost=0.45, latency=0.45, reliability=0.94, risk=0.04, asset=1.00, domain="regulated"),
    ]


def decide_task(
    task: BrainTask,
    config: BrainConfig | None = None,
    tools: list[ToolProfile] | None = None,
) -> BrainDecision:
    return ManifoldBrain(config or BrainConfig(), tools or default_tools()).decide(task)


# ---------------------------------------------------------------------------
# Phase 2.5: Asset learning — inverse RL from revealed user preferences
# ---------------------------------------------------------------------------

_SIGNAL_ASSET_VALUES: dict[str, float | None] = {
    "correction": -0.5,   # "that's not what I asked" / "wrong" → partial negative
    "acceptance": 1.0,    # "thanks" / "perfect" → full positive
    "silence": 0.3,       # no follow-up within session timeout → weak positive
    "ambiguous": None,    # skip — not enough signal to learn from
}


def classify_user_signal(
    user_response: str | None,
    *,
    no_followup: bool = False,
) -> str:
    """Classify a user response or absence of response into an asset signal.

    Parameters
    ----------
    user_response:
        The raw text string the user sent after the action, or ``None`` if the
        user did not respond at all.
    no_followup:
        ``True`` when the session timed out with no user message — treated as
        a weak positive (the user didn't complain).

    Returns
    -------
    str
        One of ``"correction"``, ``"acceptance"``, ``"silence"``, or
        ``"ambiguous"``.
    """
    if user_response is None:
        return "silence" if no_followup else "ambiguous"
    low = user_response.lower()
    if any(phrase in low for phrase in ("not what i", "wrong", "incorrect", "that's not", "that is not")):
        return "correction"
    if any(phrase in low for phrase in ("thanks", "thank you", "perfect", "great", "that worked", "correct", "exactly")):
        return "acceptance"
    if no_followup:
        return "silence"
    return "ambiguous"


@dataclass
class AssetAdapter:
    """Inverse-RL asset learner that infers true action-level A from revealed user preferences.

    ``AssetAdapter`` learns per-action asset corrections by observing whether
    users accept, correct, or silently accept the brain's responses.  The
    learning signal comes from *revealed preference* — what users actually do
    rather than explicit labels.

    Three signal types are recognised:

    * **correction** — user says the answer was wrong (e.g. "not what I asked",
      "that's wrong") → realized asset = -0.5.
    * **acceptance** — user confirms success (e.g. "thanks", "perfect") →
      realized asset = +1.0.
    * **silence** — session ended with no follow-up within timeout → realized
      asset = +0.3 (weak positive: they didn't complain).
    * **ambiguous** — any other response → skipped; no update applied.

    Asset corrections are stored per ``BrainAction`` and feed back into
    ``ManifoldBrain.expected_action_utility`` when ``asset_adapter`` is set.

    Parameters
    ----------
    lr:
        EMA learning rate for asset corrections.  Defaults to 0.12.
    min_observations:
        Minimum observations before ``adapt_asset`` applies any correction.
        Defaults to 3.

    Example
    -------
    ::

        adapter = AssetAdapter()
        brain = ManifoldBrain(config, tools=tools, asset_adapter=adapter)

        decision = brain.decide(task)
        # After the user responds:
        signal = classify_user_signal(user_response, no_followup=False)
        brain.observe_asset(decision.action, signal, task.stakes)

        # Inspect learned asset for 'clarify':
        corr = adapter.asset_corrections().get("clarify")
        _logger.debug("asset_delta: %s", corr.asset_delta)
    """

    lr: float = 0.12
    min_observations: int = 3

    _corrections: dict[str, LearnedPrices] = field(
        default_factory=dict, init=False, repr=False
    )

    def observe_outcome(
        self,
        action: "BrainAction",
        signal: str,
        stated_asset: float,
    ) -> None:
        """Update the asset correction for *action* from a user signal.

        Parameters
        ----------
        action:
            The ``BrainAction`` string that was executed.
        signal:
            One of ``"correction"``, ``"acceptance"``, ``"silence"``, or
            ``"ambiguous"``.  ``"ambiguous"`` signals are silently ignored.
        stated_asset:
            The task's ``stakes`` value at decision time — used as the
            declared asset baseline to compute the gap.
        """
        asset_realized = _SIGNAL_ASSET_VALUES.get(signal)
        if asset_realized is None:
            return
        corr = self._corrections.setdefault(action, LearnedPrices())
        gap = asset_realized - stated_asset
        corr.asset_delta = corr.asset_delta * (1.0 - self.lr) + gap * self.lr
        corr.n_observations += 1

    def adapt_asset(self, action: "BrainAction", stated_asset: float) -> float:
        """Return a corrected asset value for *action*.

        Returns *stated_asset* unchanged if fewer than ``min_observations``
        have been accumulated.

        Parameters
        ----------
        action:
            The action whose asset should be adjusted.
        stated_asset:
            The original (stated) asset value to correct.
        """
        corr = self._corrections.get(action)
        if corr is None or corr.n_observations < self.min_observations:
            return stated_asset
        return clamp01(stated_asset + corr.asset_delta)

    def asset_corrections(self) -> dict[str, LearnedPrices]:
        """Return a copy of all learned asset corrections keyed by action name."""
        return dict(self._corrections)


# ---------------------------------------------------------------------------
# Phase 3: Hierarchical decomposition — MANIFOLD of MANIFOLDs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubTaskSpec:
    """A sub-task produced by hierarchical decomposition.

    Attributes
    ----------
    prompt:
        Human-readable description of this sub-task.
    domain:
        Task domain for tool selection (inherits from parent by default).
    complexity:
        Expected complexity of this sub-task in [0, 1].
    stakes:
        Stakes for this sub-task in [0, 1]; typically ``parent.stakes * weight``.
    weight:
        Fraction of the parent asset assigned to this sub-task.  All weights
        across siblings should sum to ≤ 1.0.
    uncertainty:
        Estimated uncertainty for this sub-task.
    source_confidence:
        Inherited source confidence from parent.
    """

    prompt: str
    domain: str
    complexity: float
    stakes: float
    weight: float
    uncertainty: float = 0.5
    source_confidence: float = 0.7


@dataclass(frozen=True)
class DecompositionPlan:
    """A plan to decompose a task into ordered sub-tasks.

    Attributes
    ----------
    sub_tasks:
        Ordered tuple of sub-tasks; parent executes them in sequence.
    decompose_cost:
        The upfront cost of decomposition itself: token cost + coordination
        overhead + latency penalty.
    coordination_tax:
        Fraction of combined sub-asset lost to coordination overhead.
        A value of 0.10 means 10% of the combined child asset is deducted
        for handoff, context-passing, and synthesis.
    """

    sub_tasks: tuple[SubTaskSpec, ...]
    decompose_cost: float
    coordination_tax: float = 0.10


@dataclass(frozen=True)
class HierarchicalDecision:
    """The output of ``HierarchicalBrain.decide_hierarchical``.

    Attributes
    ----------
    decomposed:
        ``True`` when the brain chose to decompose the task.
    top_decision:
        The monolithic ``BrainDecision`` computed for the root task (always
        present — used as fallback or comparison).
    sub_decisions:
        Tuple of ``BrainDecision`` objects from child brains, or ``None`` when
        ``decomposed=False``.
    plan:
        The ``DecompositionPlan`` used, or ``None`` when not decomposed.
    combined_utility:
        The expected utility of the chosen strategy.  If decomposed, this is
        ``sum(sub_asset * weight) * (1 - tax) - decompose_cost``; otherwise
        it is ``top_decision.expected_utility``.
    depth:
        The hierarchical depth at which this decision was made (0 = root).
    """

    decomposed: bool
    top_decision: BrainDecision
    sub_decisions: tuple[BrainDecision, ...] | None
    plan: DecompositionPlan | None
    combined_utility: float
    depth: int = 0


@dataclass
class HierarchicalBrain(ManifoldBrain):
    """MANIFOLD-of-MANIFOLDs: treats task decomposition as a priced action.

    ``HierarchicalBrain`` extends ``ManifoldBrain`` with the ability to
    recursively decompose complex tasks.  Decomposition is itself a priced
    action with cost ``C_decompose``; the brain decides whether it is cheaper
    to solve the task monolithically or to break it into sub-tasks.

    Decision rule::

        if U_decompose > U_monolithic:
            decompose → spawn sub-Brains for each sub-task
        else:
            decide() normally

    where::

        U_decompose = (Σ sub_stakes * sub_weight) * (1 - coordination_tax)
                      - decompose_cost

    Parameters
    ----------
    decompose_threshold:
        Minimum task ``complexity`` before decomposition is even considered.
        Defaults to 0.72.
    max_depth:
        Maximum recursion depth.  Prevents unbounded sub-MANIFOLD creation.
        Defaults to 2.
    coordination_tax:
        Fraction of combined child asset lost to coordination overhead.
        Defaults to 0.10 (10%).

    Example
    -------
    ::

        brain = HierarchicalBrain(config, tools=tools)
        hd = brain.decide_hierarchical(task)
        if hd.decomposed:
            for sub in hd.sub_decisions:
                _logger.debug("sub decision: action=%s utility=%s", sub.action, sub.expected_utility)
        else:
            _logger.debug("top decision: %s", hd.top_decision.action)
    """

    decompose_threshold: float = 0.72
    max_depth: int = 2
    coordination_tax: float = 0.10

    def decide_hierarchical(
        self, task: BrainTask, depth: int = 0
    ) -> HierarchicalDecision:
        """Decide whether to decompose *task* or handle it monolithically.

        Parameters
        ----------
        task:
            The task to route.
        depth:
            Current recursion depth (0 for the root call).

        Returns
        -------
        HierarchicalDecision
            Contains the chosen strategy, all sub-decisions (if decomposed),
            and combined expected utility.
        """
        task = task.normalized()
        top_decision = self.decide(task)

        if task.complexity < self.decompose_threshold or depth >= self.max_depth:
            return HierarchicalDecision(
                decomposed=False,
                top_decision=top_decision,
                sub_decisions=None,
                plan=None,
                combined_utility=top_decision.expected_utility,
                depth=depth,
            )

        plan = self._make_decomposition_plan(task)
        decompose_utility = self._decomposition_utility(task, plan)

        if decompose_utility > top_decision.expected_utility:
            sub_decisions = self._execute_sub_tasks(plan, task)
            return HierarchicalDecision(
                decomposed=True,
                top_decision=top_decision,
                sub_decisions=sub_decisions,
                plan=plan,
                combined_utility=decompose_utility,
                depth=depth,
            )

        return HierarchicalDecision(
            decomposed=False,
            top_decision=top_decision,
            sub_decisions=None,
            plan=None,
            combined_utility=top_decision.expected_utility,
            depth=depth,
        )

    def _make_decomposition_plan(self, task: BrainTask) -> DecompositionPlan:
        """Heuristic 2-way split: research (60%) + synthesis (40%)."""
        sub_tasks = (
            SubTaskSpec(
                prompt=f"[Research] {task.prompt}",
                domain=task.domain,
                complexity=clamp01(task.complexity * 0.70),
                stakes=clamp01(task.stakes * 0.60),
                weight=0.60,
                uncertainty=clamp01(task.uncertainty * 0.90),
                source_confidence=task.source_confidence,
            ),
            SubTaskSpec(
                prompt=f"[Synthesize] {task.prompt}",
                domain=task.domain,
                complexity=clamp01(task.complexity * 0.50),
                stakes=clamp01(task.stakes * 0.40),
                weight=0.40,
                uncertainty=clamp01(task.uncertainty * 0.60),
                source_confidence=clamp01(task.source_confidence * 1.05),
            ),
        )
        decompose_cost = clamp01(0.08 + 0.12 * task.complexity + 0.04 * len(sub_tasks))
        return DecompositionPlan(
            sub_tasks=sub_tasks,
            decompose_cost=decompose_cost,
            coordination_tax=self.coordination_tax,
        )

    def _decomposition_utility(self, task: BrainTask, plan: DecompositionPlan) -> float:
        """Expected utility of the decomposition strategy.

        The combined value of decomposed sub-tasks approximates the full
        parent stakes minus coordination overhead and the upfront decompose
        cost.  Using parent ``task.stakes`` (not discounted sub-task stakes)
        correctly represents that the decomposition is attempting to resolve
        the same goal — just via a divide-and-conquer strategy.
        """
        combined_asset = task.stakes * (1.0 - plan.coordination_tax)
        return combined_asset - plan.decompose_cost

    def _execute_sub_tasks(
        self, plan: DecompositionPlan, parent_task: BrainTask
    ) -> tuple[BrainDecision, ...]:
        """Create child brains and run decisions for each sub-task."""
        decisions = []
        for st in plan.sub_tasks:
            child_task = BrainTask(
                prompt=st.prompt,
                domain=st.domain,
                complexity=st.complexity,
                stakes=st.stakes,
                uncertainty=st.uncertainty,
                source_confidence=st.source_confidence,
                tool_relevance=parent_task.tool_relevance,
                time_pressure=parent_task.time_pressure,
                safety_sensitivity=parent_task.safety_sensitivity,
                collaboration_value=parent_task.collaboration_value,
                user_patience=parent_task.user_patience,
            )
            child_brain = ManifoldBrain(
                config=self.config,
                tools=self.tools,
                memory=BrainMemory(),  # intentional: child brains start with clean memory (no inherited tool scars or domain stats)
                price_adapter=self.price_adapter,
            )
            decisions.append(child_brain.decide(child_task))
        return tuple(decisions)
