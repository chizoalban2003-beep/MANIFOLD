"""Phase 36: Time-Travel State Rehydration — Deterministic Decision Replay.

Implements deterministic debugging of past MANIFOLD decisions.  Given a
``task_id``, this module locates the original :class:`~manifold.provenance.DecisionReceipt`
in the :class:`~manifold.provenance.ProvenanceLedger`, reconstructs the
decision context from the stored ``grid_state_summary``, and re-runs the
:class:`~manifold.brain.ManifoldBrain` under identical conditions.

The result is a :class:`ReplayReport` that puts the **historical decision**
side-by-side with the **current decision** — making it easy to audit whether
the system has learned better behaviour since the original task was run.

Key classes
-----------
``ReplayReport``
    Side-by-side audit record: historical receipt vs. current re-decision.
``StateRehydrator``
    Locates a :class:`~manifold.provenance.DecisionReceipt` and orchestrates
    a :class:`VirtualExecution`.
``VirtualExecution``
    Re-instantiates a :class:`~manifold.brain.ManifoldBrain` with the
    parameters stored in a :class:`~manifold.provenance.DecisionReceipt` and
    produces a new :class:`~manifold.brain.BrainDecision`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .brain import BrainDecision, BrainTask, ManifoldBrain
from .provenance import DecisionReceipt, ProvenanceLedger


# ---------------------------------------------------------------------------
# ReplayReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayReport:
    """Side-by-side audit of a historical decision vs. a current re-decision.

    Attributes
    ----------
    task_id:
        The task identifier that was replayed.
    found:
        ``True`` if a :class:`~manifold.provenance.DecisionReceipt` was found
        for *task_id*.  If ``False``, all other fields except *task_id* are
        empty / default.
    historical_receipt:
        The original :class:`~manifold.provenance.DecisionReceipt`, or
        ``None`` if not found.
    historical_action:
        The action string from the original decision (``"use_tool"``, etc.).
    current_action:
        The action string produced by re-running the brain now.
    current_risk_score:
        Risk score from the re-run decision.
    current_confidence:
        Confidence from the re-run decision.
    action_changed:
        ``True`` if *historical_action* differs from *current_action*.
    replay_timestamp:
        POSIX timestamp when the replay was performed.
    notes:
        Human-readable audit notes explaining any differences.
    """

    task_id: str
    found: bool
    historical_receipt: DecisionReceipt | None
    historical_action: str
    current_action: str
    current_risk_score: float
    current_confidence: float
    action_changed: bool
    replay_timestamp: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "task_id": self.task_id,
            "found": self.found,
            "historical_action": self.historical_action,
            "current_action": self.current_action,
            "current_risk_score": round(self.current_risk_score, 4),
            "current_confidence": round(self.current_confidence, 4),
            "action_changed": self.action_changed,
            "replay_timestamp": self.replay_timestamp,
            "notes": list(self.notes),
            "historical_receipt": (
                self.historical_receipt.to_dict()
                if self.historical_receipt is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# VirtualExecution
# ---------------------------------------------------------------------------


@dataclass
class VirtualExecution:
    """Re-runs a :class:`~manifold.brain.ManifoldBrain` from historical parameters.

    Given a :class:`~manifold.provenance.DecisionReceipt`, this class
    reconstructs a :class:`~manifold.brain.BrainTask` from the stored
    ``grid_state_summary`` and calls a :class:`~manifold.brain.ManifoldBrain`
    to produce a new decision.

    Parameters
    ----------
    brain:
        A :class:`~manifold.brain.ManifoldBrain` instance to use for
        re-evaluation.  The brain is called with a reconstructed task derived
        from the receipt's ``grid_state_summary``.
    """

    brain: ManifoldBrain

    def execute(self, receipt: DecisionReceipt) -> BrainDecision:
        """Re-run the brain using parameters extracted from *receipt*.

        Parameters
        ----------
        receipt:
            The original :class:`~manifold.provenance.DecisionReceipt` to
            replay.

        Returns
        -------
        BrainDecision
            A new decision produced under current brain state with the
            historical task parameters.
        """
        task = _task_from_receipt(receipt)
        return self.brain.decide(task)


def _task_from_receipt(receipt: DecisionReceipt) -> BrainTask:
    """Reconstruct a :class:`~manifold.brain.BrainTask` from a receipt's summary.

    The ``grid_state_summary`` stores the brain's decision output, not the
    original input task.  We therefore invert the available signals to
    approximate the original task parameters.  Fields not recorded in the
    summary fall back to neutral defaults.
    """
    summary = receipt.grid_state_summary
    return BrainTask(
        prompt=f"replay:{receipt.task_id}",
        domain=str(summary.get("domain", "general")),
        uncertainty=float(summary.get("uncertainty", 0.5)),
        complexity=float(summary.get("complexity", 0.5)),
        stakes=float(summary.get("stakes", 0.5)),
        source_confidence=float(summary.get("source_confidence", 0.7)),
        tool_relevance=float(summary.get("tool_relevance", 0.5)),
        time_pressure=float(summary.get("time_pressure", 0.4)),
        safety_sensitivity=float(summary.get("safety_sensitivity", 0.2)),
        collaboration_value=float(summary.get("collaboration_value", 0.3)),
        user_patience=float(summary.get("user_patience", 0.7)),
    )


# ---------------------------------------------------------------------------
# StateRehydrator
# ---------------------------------------------------------------------------


@dataclass
class StateRehydrator:
    """Orchestrates time-travel replay of a past MANIFOLD decision.

    Parameters
    ----------
    ledger:
        :class:`~manifold.provenance.ProvenanceLedger` to query for the
        original :class:`~manifold.provenance.DecisionReceipt`.
    brain:
        A :class:`~manifold.brain.ManifoldBrain` instance used to re-run
        the decision under current conditions.  If ``None``, a default brain
        is instantiated.

    Example
    -------
    ::

        rehydrator = StateRehydrator(ledger=provenance_ledger, brain=brain)
        report = rehydrator.replay("task-abc123")
        if report.action_changed:
            print("The system has learned better behaviour!")
    """

    ledger: ProvenanceLedger
    brain: ManifoldBrain = field(default_factory=lambda: ManifoldBrain(tools=[]))
    _clock: Any = field(default=None, init=False, repr=False)

    def replay(self, task_id: str) -> ReplayReport:
        """Locate the original receipt for *task_id* and produce a :class:`ReplayReport`.

        Parameters
        ----------
        task_id:
            The task identifier to look up and replay.

        Returns
        -------
        ReplayReport
            Side-by-side comparison of the historical and current decision.
            If *task_id* is not found, :attr:`ReplayReport.found` is
            ``False`` and all decision fields are empty defaults.
        """
        ts = self._clock() if callable(self._clock) else time.time()

        receipt = self.ledger.get(task_id)

        if receipt is None:
            return ReplayReport(
                task_id=task_id,
                found=False,
                historical_receipt=None,
                historical_action="",
                current_action="",
                current_risk_score=0.0,
                current_confidence=0.0,
                action_changed=False,
                replay_timestamp=ts,
                notes=("No provenance receipt found for this task_id.",),
            )

        # Re-run the brain with the historical parameters
        executor = VirtualExecution(brain=self.brain)
        current_decision = executor.execute(receipt)

        historical_action = receipt.final_decision
        current_action = current_decision.action
        action_changed = historical_action != current_action

        notes_list: list[str] = []
        if action_changed:
            notes_list.append(
                f"Action changed: {historical_action!r} → {current_action!r}. "
                "System behaviour may have evolved since the original decision."
            )
        else:
            notes_list.append(
                f"Action unchanged: {current_action!r}. "
                "Decision is consistent with historical behaviour."
            )

        hist_risk = receipt.grid_state_summary.get("risk_score", 0.0)
        if isinstance(hist_risk, (int, float)):
            risk_delta = current_decision.risk_score - float(hist_risk)
            if abs(risk_delta) > 0.1:
                direction = "higher" if risk_delta > 0 else "lower"
                notes_list.append(
                    f"Risk score is {direction} now "
                    f"({current_decision.risk_score:.4f} vs historical {float(hist_risk):.4f})."
                )

        return ReplayReport(
            task_id=task_id,
            found=True,
            historical_receipt=receipt,
            historical_action=historical_action,
            current_action=current_action,
            current_risk_score=current_decision.risk_score,
            current_confidence=current_decision.confidence,
            action_changed=action_changed,
            replay_timestamp=ts,
            notes=tuple(notes_list),
        )

    def replay_all(self) -> list[ReplayReport]:
        """Replay every decision recorded in the ledger.

        Returns
        -------
        list[ReplayReport]
            One :class:`ReplayReport` per receipt, in ledger order.
        """
        return [self.replay(r.task_id) for r in self.ledger.all_receipts()]
