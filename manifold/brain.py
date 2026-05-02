"""MANIFOLD Brain: generalized adaptive decision layer for agents.

MANIFOLD Brain is the product-facing meta-controller above GridMapper OS and
TrustRouter. It chooses the next action for an AI agent, tool-using workflow, or
human-in-the-loop system by pricing the current task as cost, risk, asset, and
rule pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Callable, Literal

from .gridmapper import AgentPopulation, GridOptimizationResult, GridWorld
from .trustrouter import clamp01


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

    @property
    def utility(self) -> float:
        return self.asset_gained - self.cost_paid - self.risk_realized - self.rule_violations


@dataclass
class BrainMemory:
    """Cross-world memory for domains, actions, and tools."""

    domain_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    action_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    tool_stats: dict[str, dict[str, float]] = field(default_factory=dict)

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
        self._update_bucket(self.domain_stats, task.domain, outcome)
        self._update_bucket(self.action_stats, decision.action, outcome)
        if decision.selected_tool:
            self._update_bucket(self.tool_stats, decision.selected_tool, outcome)

    def _update_bucket(
        self, store: dict[str, dict[str, float]], key: str, outcome: BrainOutcome
    ) -> None:
        current = store.get(key, {"count": 0.0, "success_rate": 1.0, "utility": 0.0})
        count = current["count"]
        new_count = count + 1.0
        current["success_rate"] = (
            current["success_rate"] * count + (1.0 if outcome.success else 0.0)
        ) / new_count
        current["utility"] = (current["utility"] * count + outcome.utility) / new_count
        current["count"] = new_count
        store[key] = current


@dataclass
class ManifoldBrain:
    """Adaptive meta-controller for agent decisions."""

    config: BrainConfig = field(default_factory=BrainConfig)
    tools: list[ToolProfile] = field(default_factory=list)
    memory: BrainMemory = field(default_factory=BrainMemory)

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
            reliability = clamp01(tool.reliability + self.memory.tool_reliability_adjustment(tool))
            utility = tool.asset * reliability - tool.cost - tool.latency - tool.risk
            adjusted.append((utility, tool))
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
        action_asset = task.stakes + tool_bonus
        risk_discount = risk_score * brain_action_risk_multiplier(action)
        return action_asset - cost - risk_discount

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
