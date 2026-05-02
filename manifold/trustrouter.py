"""TrustRouter: MANIFOLD's first niche product layer.

TrustRouter maps an AI-agent or dialogue task into GridMapper OS, runs the
evolved MANIFOLD substrate, and returns an action policy:

- answer directly,
- clarify,
- retrieve,
- verify,
- escalate,
- refuse.

The goal is not to hard-code "safe" behavior. The goal is to price the task:
tokens, latency, user patience, hallucination loss, source confidence, and
safety sensitivity. MANIFOLD then compiles those prices into thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Literal

from .gridmapper import AgentPopulation, GridOptimizationResult, GridWorld


TrustAction = Literal["answer", "clarify", "retrieve", "verify", "escalate", "refuse"]


@dataclass(frozen=True)
class DialogueTask:
    """A task-level description that can be projected into MANIFOLD."""

    prompt: str
    domain: str = "general"
    uncertainty: float = 0.5
    complexity: float = 0.5
    stakes: float = 0.5
    source_confidence: float = 0.7
    user_patience: float = 0.7
    safety_sensitivity: float = 0.2
    dynamic_intent: bool = False

    def normalized(self) -> "DialogueTask":
        return DialogueTask(
            prompt=self.prompt,
            domain=self.domain,
            uncertainty=clamp01(self.uncertainty),
            complexity=clamp01(self.complexity),
            stakes=clamp01(self.stakes),
            source_confidence=clamp01(self.source_confidence),
            user_patience=clamp01(self.user_patience),
            safety_sensitivity=clamp01(self.safety_sensitivity),
            dynamic_intent=self.dynamic_intent,
        )


@dataclass(frozen=True)
class TrustRouterConfig:
    """Execution costs for dialogue policy compilation."""

    grid_size: int = 11
    generations: int = 40
    population_size: int = 64
    predators: float = 0.05
    seed: int = 2500
    token_cost_weight: float = 0.08
    latency_cost_weight: float = 0.06
    clarification_cost_weight: float = 0.22
    retrieval_cost_weight: float = 0.16
    human_escalation_cost: float = 0.65


@dataclass(frozen=True)
class TrustRouterDecision:
    """Compiled policy for one dialogue task."""

    action: TrustAction
    confidence: float
    risk_score: float
    verification_threshold: float
    clarification_threshold: float
    retrieval_threshold: float
    escalation_threshold: float
    reputation_cap: float
    robustness_score: float
    recommended_verification_rate: float
    recommended_gossip_rate: float
    notes: tuple[str, ...]
    result: GridOptimizationResult


@dataclass
class TrustLearningMemory:
    """Small cross-world memory for learned policy thresholds.

    This is intentionally simple and serializable: it stores rolling averages by
    domain. It gives the system a memory of prior worlds without hiding policy in
    hard-coded rules.
    """

    domain_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def prior_adjustment(self, task: DialogueTask) -> float:
        stats = self.domain_stats.get(task.domain)
        if not stats:
            return 0.0
        # Positive adjustment means prior worlds were risky and need more care.
        return clamp(stats.get("risk", 0.0) - stats.get("robustness", 1.0), -0.2, 0.2)

    def update(self, task: DialogueTask, decision: TrustRouterDecision) -> None:
        existing = self.domain_stats.get(task.domain)
        sample = {
            "risk": decision.risk_score,
            "verification": decision.recommended_verification_rate,
            "reputation_cap": decision.reputation_cap,
            "robustness": decision.robustness_score,
            "count": 1.0,
        }
        if not existing:
            self.domain_stats[task.domain] = sample
            return
        count = existing.get("count", 1.0)
        new_count = count + 1.0
        for key in ("risk", "verification", "reputation_cap", "robustness"):
            existing[key] = (existing.get(key, sample[key]) * count + sample[key]) / new_count
        existing["count"] = new_count


@dataclass
class TrustRouter:
    """Compile dialogue tasks into adaptive MANIFOLD policies."""

    config: TrustRouterConfig = field(default_factory=TrustRouterConfig)
    memory: TrustLearningMemory = field(default_factory=TrustLearningMemory)

    def route(self, task: DialogueTask) -> TrustRouterDecision:
        task = task.normalized()
        world = self.map_task_to_world(task)
        verification_cost = self._verification_cost(task)
        false_trust_penalty = self._false_trust_penalty(task)
        result = AgentPopulation(
            seed=str(self.config.seed),
            n=self.config.population_size,
            predators=self.config.predators,
        ).optimize(
            world,
            generations=self.config.generations,
            verification_cost=verification_cost,
            false_trust_penalty=false_trust_penalty,
        )
        risk_score = self._risk_score(task)
        risk_score = clamp01(risk_score + self.memory.prior_adjustment(task))
        action = self._choose_action(task, result, risk_score)
        decision = TrustRouterDecision(
            action=action,
            confidence=clamp01(1.0 - risk_score + result.audit.robustness_score * 0.25),
            risk_score=risk_score,
            verification_threshold=result.audit.verification_threshold,
            clarification_threshold=self._clarification_threshold(task),
            retrieval_threshold=self._retrieval_threshold(task),
            escalation_threshold=self._escalation_threshold(task),
            reputation_cap=result.reputation_cap,
            robustness_score=result.audit.robustness_score,
            recommended_verification_rate=result.verification,
            recommended_gossip_rate=result.gossip,
            notes=self._notes(task, result, risk_score),
            result=result,
        )
        self.memory.update(task, decision)
        return decision

    def map_task_to_world(self, task: DialogueTask) -> GridWorld:
        """Project dialogue uncertainty into a `[C,R,N,A]` world."""

        size = self.config.grid_size
        center = size // 2
        world = GridWorld(size=size, seed=self.config.seed)
        for row in range(size):
            for col in range(size):
                complexity = abs(row - center) / max(1, center)
                ambiguity = abs(col - center) / max(1, center)
                cost = (
                    0.02
                    + self.config.token_cost_weight * complexity
                    + self.config.latency_cost_weight * task.complexity
                )
                risk = clamp01(
                    0.08
                    + 0.45 * ambiguity
                    + 0.30 * task.uncertainty
                    + 0.30 * (1.0 - task.source_confidence)
                    + 0.25 * task.safety_sensitivity
                )
                neutrality = clamp01(0.9 - 0.4 * complexity - 0.3 * ambiguity)
                asset = 0.05 + 0.25 * task.stakes * (1.0 - ambiguity)
                world.set_cell(row, col, cost=cost, risk=risk, neutrality=neutrality, asset=asset)

        target_pos = (
            center,
            max(0, min(size - 1, round(center + (task.uncertainty - 0.5) * center))),
        )
        world.add_dynamic_targets(
            [
                {
                    "id": "resolved_intent",
                    "pos": target_pos,
                    "asset": 2.0 + task.stakes * 6.0,
                    "moves": "random_walk" if task.dynamic_intent else "static",
                }
            ]
        )
        world.add_rule("hallucination", penalty=2.0 + task.stakes * 8.0, triggers="trusted_lie")
        world.add_rule("wrong_assumption", penalty=1.0 + task.uncertainty * 4.0, triggers="miss_target")
        world.add_rule("low_confidence_exhaustion", penalty=1.0, triggers="low_energy")
        return world

    def _choose_action(
        self, task: DialogueTask, result: GridOptimizationResult, risk_score: float
    ) -> TrustAction:
        if (
            risk_score <= 0.25
            and task.source_confidence >= 0.8
            and task.safety_sensitivity <= 0.35
            and not task.dynamic_intent
        ):
            return "answer"
        if task.safety_sensitivity >= 0.85 and risk_score >= 0.75:
            return "refuse"
        if risk_score >= self._escalation_threshold(task):
            return "escalate"
        if task.uncertainty >= self._clarification_threshold(task) and task.user_patience >= 0.35:
            return "clarify"
        if task.source_confidence >= 0.55 and risk_score >= result.audit.verification_threshold:
            return "verify"
        if (1.0 - task.source_confidence) >= self._retrieval_threshold(task):
            return "retrieve"
        if risk_score >= result.audit.verification_threshold:
            return "verify"
        return "answer"

    def _risk_score(self, task: DialogueTask) -> float:
        return clamp01(
            0.35 * task.uncertainty
            + 0.25 * task.complexity
            + 0.25 * task.stakes
            + 0.25 * (1.0 - task.source_confidence)
            + 0.35 * task.safety_sensitivity
            + (0.10 if task.dynamic_intent else 0.0)
        )

    def _verification_cost(self, task: DialogueTask) -> float:
        return max(0.02, self.config.retrieval_cost_weight * (1.1 - task.user_patience))

    def _false_trust_penalty(self, task: DialogueTask) -> float:
        return 0.5 + 3.0 * task.stakes + 2.0 * task.safety_sensitivity

    def _clarification_threshold(self, task: DialogueTask) -> float:
        return clamp01(0.65 - 0.25 * task.stakes + 0.20 * (1.0 - task.user_patience))

    def _retrieval_threshold(self, task: DialogueTask) -> float:
        return clamp01(0.45 - 0.20 * task.stakes - 0.20 * task.safety_sensitivity)

    def _escalation_threshold(self, task: DialogueTask) -> float:
        return clamp01(0.82 - 0.25 * task.safety_sensitivity - 0.15 * task.stakes)

    def _notes(
        self, task: DialogueTask, result: GridOptimizationResult, risk_score: float
    ) -> tuple[str, ...]:
        notes = [
            f"Mapped task '{task.domain}' into GridMapper world with risk={risk_score:.2f}.",
            f"Recommended verification rate is {result.verification:.1%}.",
            f"Reputation cap settled near {result.reputation_cap:.2f}.",
        ]
        if task.dynamic_intent:
            notes.append("Dynamic intent raised target volatility; clarification becomes cheaper earlier.")
        if task.safety_sensitivity > 0.7:
            notes.append("High safety sensitivity makes escalation/refusal economically plausible.")
        if result.audit.monopoly_risk > 0.05:
            notes.append("Tool/source reputation concentration detected; keep Predatory Scouts active.")
        return tuple(notes)


def route_task(task: DialogueTask, config: TrustRouterConfig | None = None) -> TrustRouterDecision:
    """Convenience API for one-shot TrustRouter decisions."""

    return TrustRouter(config or TrustRouterConfig()).route(task)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def average_decision_risk(decisions: list[TrustRouterDecision]) -> float:
    if not decisions:
        return 0.0
    return fmean(decision.risk_score for decision in decisions)
