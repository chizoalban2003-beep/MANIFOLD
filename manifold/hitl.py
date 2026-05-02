"""Phase 9: Human-in-the-Loop (HITL) Governor.

When MANIFOLD's ``ManifoldBrain`` escalates a task, or when the estimated
``Risk × Stakes`` product exceeds an operator-set threshold, the
``HITLGate`` intercepts the decision and routes it to a human reviewer.

The key insight: **the human is priced as a tool.**  A human reviewer has
a stated cost (time/salary), a latency (response time), and a reliability
(accuracy).  MANIFOLD will learn, over time, to only call for a human when
the **Cost of Error** outweighs the **Cost of Human Intervention.**

When a human provides feedback, it is ingested as a **TeacherSpike** — an
indelible super-scar that overrides all other gossip and leaves a permanent
high-weight imprint in ``BrainMemory``.  Teacher spikes represent ground
truth in high-stakes situations.

Key classes
-----------
``HITLRecord``
    A single escalation event: the task, brain decision, escalation reason,
    and optional human verdict.
``HITLConfig``
    Thresholds controlling when the HITL gate triggers.
``HITLGate``
    Intercepts ``BrainDecision`` objects whose risk × stakes exceeds the
    threshold and queues them for human review.
``TeacherSpike``
    An indelible high-weight correction to ``BrainMemory``, injected when
    human feedback is provided after an escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .brain import BrainDecision, BrainMemory, BrainOutcome, BrainTask, GossipNote
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# HITLRecord
# ---------------------------------------------------------------------------


@dataclass
class HITLRecord:
    """A single HITL escalation event.

    Attributes
    ----------
    task:
        The ``BrainTask`` that was escalated.
    decision:
        The ``BrainDecision`` that triggered escalation.
    reason:
        Short description of why the escalation was triggered.
    risk_stakes_product:
        The ``risk_score × stakes`` value that exceeded the threshold.
    resolved:
        Whether a human has provided a verdict.
    human_verdict:
        ``"approve"``, ``"reject"``, or ``"correct"`` after human review.
    human_correction:
        Free-text correction or revised action from the human (optional).
    """

    task: BrainTask
    decision: BrainDecision
    reason: str
    risk_stakes_product: float
    resolved: bool = False
    human_verdict: str = ""
    human_correction: str = ""


# ---------------------------------------------------------------------------
# HITLConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HITLConfig:
    """Configuration for the ``HITLGate``.

    Parameters
    ----------
    risk_stakes_threshold:
        Risk × Stakes product above which escalation is forced.
        Default: 0.55 (i.e., risk=0.7 × stakes=0.8 = 0.56 → escalate).
    force_escalate_actions:
        Set of ``BrainAction`` strings that always trigger HITL regardless
        of risk score.  Defaults to ``{"refuse"}``.
    human_cost:
        Priced cost of a human review call (default: 0.45 — matches
        ``human_reviewer`` ToolProfile).
    human_latency:
        Expected latency of a human response in normalised units (default: 0.45).
    human_reliability:
        Human review reliability prior (default: 0.94).
    teacher_spike_weight:
        EMA weight applied to a TeacherSpike update.  Higher = stronger scar.
        Default: 0.85 (much stronger than the standard 0.15 learning rate).
    """

    risk_stakes_threshold: float = 0.55
    force_escalate_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({"refuse"})
    )
    human_cost: float = 0.45
    human_latency: float = 0.45
    human_reliability: float = 0.94
    teacher_spike_weight: float = 0.85


# ---------------------------------------------------------------------------
# TeacherSpike
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherSpike:
    """An indelible human-override correction for ``BrainMemory``.

    A ``TeacherSpike`` is injected when a human provides feedback after a
    HITL escalation.  It writes a super-scar with a very high learning rate
    (default: 0.85) that overrides accumulated gossip and tool stats.

    Unlike ordinary outcome-based learning (LR ≈ 0.15), a ``TeacherSpike``
    represents a high-confidence ground-truth signal.  The system will not
    "forget" it quickly because the delta is large and the weight is high.

    Attributes
    ----------
    tool_name:
        The tool (or domain) the teacher correction applies to.
    verdict:
        ``"approve"`` (human confirmed the action was correct),
        ``"reject"`` (human rejected the action), or
        ``"correct"`` (human provided a different action/answer).
    confidence:
        Human reviewer's confidence in their verdict [0, 1].  Default: 0.90.
    """

    tool_name: str
    verdict: Literal["approve", "reject", "correct"]
    confidence: float = 0.90

    def apply_to_memory(
        self,
        memory: BrainMemory,
        spike_weight: float = 0.85,
    ) -> None:
        """Apply the teacher spike as a super-scar to *memory*.

        Parameters
        ----------
        memory:
            The ``BrainMemory`` to update.
        spike_weight:
            The high-weight EMA learning rate for this update.  Defaults to
            ``HITLConfig.teacher_spike_weight`` (0.85).

        Effect:
        - ``"approve"``: Raises success_rate toward 1.0 with spike weight.
        - ``"reject"``/``"correct"``: Lowers success_rate toward 0.0 with spike weight.
        """
        stats = memory.tool_stats.setdefault(
            self.tool_name,
            {
                "success_rate": 1.0,
                "count": 0.0,
                "utility": 0.5,
                "consecutive_failures": 0.0,
            },
        )
        current = stats["success_rate"]
        if self.verdict == "approve":
            target = 1.0
        else:
            target = 0.0

        # Super-scar EMA: delta × confidence × spike_weight
        effective_weight = spike_weight * self.confidence
        stats["success_rate"] = clamp01(
            current * (1.0 - effective_weight) + target * effective_weight
        )
        stats["count"] = stats.get("count", 0.0) + 1.0

        if self.verdict in {"reject", "correct"}:
            stats["consecutive_failures"] = stats.get("consecutive_failures", 0.0) + 1.0


# ---------------------------------------------------------------------------
# HITLGate
# ---------------------------------------------------------------------------


@dataclass
class HITLGate:
    """Human-in-the-loop escalation gate for high-risk MANIFOLD decisions.

    ``HITLGate`` sits between the ``ManifoldBrain`` decision and physical
    execution.  It intercepts any decision where ``risk_score × task.stakes``
    exceeds ``config.risk_stakes_threshold`` and queues it for human review.

    The human is priced as a tool — MANIFOLD learns over time that human
    review costs 0.45 per call, and will only escalate when the potential
    cost of error exceeds that budget.

    Parameters
    ----------
    config:
        ``HITLConfig`` instance controlling thresholds and weights.
    memory:
        The ``BrainMemory`` that teacher spikes will be applied to.

    Example
    -------
    ::

        gate = HITLGate(config=HITLConfig(risk_stakes_threshold=0.50), memory=brain.memory)

        decision = brain.decide(high_stakes_task)
        if gate.should_escalate(high_stakes_task, decision):
            record = gate.escalate(high_stakes_task, decision)
            # ... route to human review UI ...
            gate.resolve(record, verdict="reject", correction="Use a safer tool")
            # → TeacherSpike applied to BrainMemory
    """

    config: HITLConfig = field(default_factory=HITLConfig)
    memory: BrainMemory = field(default_factory=BrainMemory)

    _queue: list[HITLRecord] = field(default_factory=list, init=False, repr=False)
    _resolved: list[HITLRecord] = field(default_factory=list, init=False, repr=False)
    _spike_history: list[TeacherSpike] = field(default_factory=list, init=False, repr=False)

    def should_escalate(self, task: BrainTask, decision: BrainDecision) -> bool:
        """Return ``True`` if this task+decision should be sent to human review.

        Triggers escalation when:
        1. ``decision.action`` is in ``config.force_escalate_actions``, OR
        2. ``decision.risk_score × task.stakes >= config.risk_stakes_threshold``.

        Parameters
        ----------
        task:
            The current task.
        decision:
            The brain's decision for the task.
        """
        if decision.action in self.config.force_escalate_actions:
            return True
        rsp = decision.risk_score * task.stakes
        return rsp >= self.config.risk_stakes_threshold

    def escalate(self, task: BrainTask, decision: BrainDecision) -> HITLRecord:
        """Queue a task for human review and return the escalation record.

        Parameters
        ----------
        task:
            The task being escalated.
        decision:
            The brain decision that triggered escalation.

        Returns
        -------
        HITLRecord
            The queued escalation record.  Resolve it via ``resolve()``.
        """
        rsp = decision.risk_score * task.stakes
        if decision.action in self.config.force_escalate_actions:
            reason = f"Action '{decision.action}' requires human review (force_escalate policy)"
        else:
            reason = (
                f"risk×stakes={rsp:.3f} ≥ threshold={self.config.risk_stakes_threshold:.3f}"
            )
        record = HITLRecord(
            task=task,
            decision=decision,
            reason=reason,
            risk_stakes_product=rsp,
        )
        self._queue.append(record)
        return record

    def resolve(
        self,
        record: HITLRecord,
        verdict: Literal["approve", "reject", "correct"],
        correction: str = "",
        tool_name: str = "",
        confidence: float = 0.90,
    ) -> TeacherSpike:
        """Resolve an escalation with a human verdict and apply a TeacherSpike.

        Parameters
        ----------
        record:
            The ``HITLRecord`` to resolve.
        verdict:
            Human verdict: ``"approve"``, ``"reject"``, or ``"correct"``.
        correction:
            Optional free-text correction or revised action.
        tool_name:
            The tool to apply the spike to.  Defaults to the task domain.
        confidence:
            Human reviewer's confidence (default: 0.90).

        Returns
        -------
        TeacherSpike
            The spike that was applied to memory.
        """
        record.resolved = True
        record.human_verdict = verdict
        record.human_correction = correction
        if record in self._queue:
            self._queue.remove(record)
        self._resolved.append(record)

        spike_target = tool_name or record.task.domain
        spike = TeacherSpike(
            tool_name=spike_target,
            verdict=verdict,
            confidence=confidence,
        )
        spike.apply_to_memory(self.memory, spike_weight=self.config.teacher_spike_weight)
        self._spike_history.append(spike)
        return spike

    def pending_queue(self) -> list[HITLRecord]:
        """Return all unresolved escalation records."""
        return [r for r in self._queue if not r.resolved]

    def resolved_records(self) -> list[HITLRecord]:
        """Return all resolved escalation records."""
        return list(self._resolved)

    def spike_history(self) -> list[TeacherSpike]:
        """Return all TeacherSpikes that have been applied."""
        return list(self._spike_history)

    def escalation_count(self) -> int:
        """Total escalations (pending + resolved)."""
        return len(self._queue) + len(self._resolved)

    def rejection_rate(self) -> float:
        """Fraction of resolved escalations that were rejected or corrected."""
        if not self._resolved:
            return 0.0
        rejects = sum(
            1 for r in self._resolved if r.human_verdict in {"reject", "correct"}
        )
        return rejects / len(self._resolved)

    def escalation_cost(self) -> float:
        """Total priced cost of all escalations (resolved + pending).

        Each escalation costs ``config.human_cost``.
        """
        return self.escalation_count() * self.config.human_cost

    def to_brain_outcome(self, record: HITLRecord) -> BrainOutcome:
        """Convert a resolved escalation record to a ``BrainOutcome``.

        Parameters
        ----------
        record:
            A resolved ``HITLRecord``.

        Returns
        -------
        BrainOutcome
            With cost set to ``config.human_cost`` and success based on verdict.
        """
        success = record.human_verdict == "approve"
        return BrainOutcome(
            success=success,
            cost_paid=self.config.human_cost,
            risk_realized=0.0 if success else 0.30,
            asset_gained=1.0 if success else 0.0,
            failure_mode="" if success else "tool_error",
        )
