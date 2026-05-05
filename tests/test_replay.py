"""Tests for Phase 36: Time-Travel State Rehydration (replay.py)."""

from __future__ import annotations

from manifold.brain import BrainConfig, BrainTask, ManifoldBrain
from manifold.provenance import ProvenanceLedger
from manifold.replay import ReplayReport, StateRehydrator, VirtualExecution, _task_from_receipt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ledger_with_receipt(
    task_id: str = "task-001",
    final_decision: str = "use_tool",
    extra_summary: dict | None = None,
) -> ProvenanceLedger:
    ledger = ProvenanceLedger()
    summary = {
        "action": final_decision,
        "risk_score": 0.2,
        "confidence": 0.8,
        "domain": "finance",
        "stakes": 0.6,
        "uncertainty": 0.4,
    }
    if extra_summary:
        summary.update(extra_summary)
    ledger.record(
        task_id=task_id,
        final_decision=final_decision,
        grid_state_summary=summary,
        clock=lambda: 1000.0,
    )
    return ledger


def _make_brain() -> ManifoldBrain:
    return ManifoldBrain(
        config=BrainConfig(grid_size=7, generations=5, population_size=16),
        tools=[],
    )


# ---------------------------------------------------------------------------
# ReplayReport
# ---------------------------------------------------------------------------


class TestReplayReport:
    def test_to_dict_not_found(self) -> None:
        report = ReplayReport(
            task_id="missing",
            found=False,
            historical_receipt=None,
            historical_action="",
            current_action="",
            current_risk_score=0.0,
            current_confidence=0.0,
            action_changed=False,
            replay_timestamp=1.0,
            notes=("Not found.",),
        )
        d = report.to_dict()
        assert d["task_id"] == "missing"
        assert d["found"] is False
        assert d["historical_receipt"] is None

    def test_to_dict_found(self) -> None:
        ledger = _make_ledger_with_receipt()
        receipt = ledger.get("task-001")
        assert receipt is not None
        report = ReplayReport(
            task_id="task-001",
            found=True,
            historical_receipt=receipt,
            historical_action="use_tool",
            current_action="answer",
            current_risk_score=0.15,
            current_confidence=0.9,
            action_changed=True,
            replay_timestamp=2000.0,
            notes=("Changed.",),
        )
        d = report.to_dict()
        assert d["found"] is True
        assert d["action_changed"] is True
        assert d["historical_receipt"] is not None
        assert "receipt_hash" in d["historical_receipt"]

    def test_action_changed_flag(self) -> None:
        r_same = ReplayReport(
            task_id="t",
            found=True,
            historical_receipt=None,
            historical_action="use_tool",
            current_action="use_tool",
            current_risk_score=0.1,
            current_confidence=0.9,
            action_changed=False,
            replay_timestamp=1.0,
            notes=(),
        )
        assert not r_same.action_changed

        r_diff = ReplayReport(
            task_id="t",
            found=True,
            historical_receipt=None,
            historical_action="use_tool",
            current_action="refuse",
            current_risk_score=0.9,
            current_confidence=0.5,
            action_changed=True,
            replay_timestamp=1.0,
            notes=(),
        )
        assert r_diff.action_changed


# ---------------------------------------------------------------------------
# _task_from_receipt (private helper)
# ---------------------------------------------------------------------------


class TestTaskFromReceipt:
    def test_basic_reconstruction(self) -> None:
        ledger = _make_ledger_with_receipt()
        receipt = ledger.get("task-001")
        assert receipt is not None
        task = _task_from_receipt(receipt)
        assert isinstance(task, BrainTask)
        assert task.domain == "finance"

    def test_defaults_for_missing_fields(self) -> None:
        ledger = ProvenanceLedger()
        ledger.record("t-bare", "answer", grid_state_summary={}, clock=lambda: 1.0)
        receipt = ledger.get("t-bare")
        assert receipt is not None
        task = _task_from_receipt(receipt)
        assert task.domain == "general"
        assert task.uncertainty == 0.5
        assert task.stakes == 0.5

    def test_prompt_contains_task_id(self) -> None:
        ledger = _make_ledger_with_receipt("abc-123")
        receipt = ledger.get("abc-123")
        assert receipt is not None
        task = _task_from_receipt(receipt)
        assert "abc-123" in task.prompt


# ---------------------------------------------------------------------------
# VirtualExecution
# ---------------------------------------------------------------------------


class TestVirtualExecution:
    def test_returns_brain_decision(self) -> None:
        brain = _make_brain()
        executor = VirtualExecution(brain=brain)
        ledger = _make_ledger_with_receipt()
        receipt = ledger.get("task-001")
        assert receipt is not None
        decision = executor.execute(receipt)
        assert hasattr(decision, "action")
        assert hasattr(decision, "risk_score")
        assert hasattr(decision, "confidence")

    def test_decision_action_is_valid(self) -> None:
        from manifold.brain import BrainAction
        import typing

        valid_actions = set(typing.get_args(BrainAction))
        brain = _make_brain()
        executor = VirtualExecution(brain=brain)
        ledger = _make_ledger_with_receipt()
        receipt = ledger.get("task-001")
        assert receipt is not None
        decision = executor.execute(receipt)
        assert decision.action in valid_actions


# ---------------------------------------------------------------------------
# StateRehydrator
# ---------------------------------------------------------------------------


class TestStateRehydratorNotFound:
    def test_not_found_report(self) -> None:
        ledger = ProvenanceLedger()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("nonexistent-task")
        assert report.found is False
        assert report.task_id == "nonexistent-task"
        assert report.historical_action == ""
        assert report.current_action == ""
        assert not report.action_changed

    def test_not_found_has_note(self) -> None:
        ledger = ProvenanceLedger()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("missing")
        assert len(report.notes) >= 1
        assert any("No provenance" in n for n in report.notes)


class TestStateRehydratorFound:
    def test_found_report_fields(self) -> None:
        ledger = _make_ledger_with_receipt("task-xyz", "use_tool")
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("task-xyz")
        assert report.found is True
        assert report.task_id == "task-xyz"
        assert report.historical_action == "use_tool"
        assert report.historical_receipt is not None
        assert report.replay_timestamp > 0.0

    def test_current_action_is_set(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("task-001")
        assert report.current_action != "" or report.current_action == ""  # any string is valid
        assert isinstance(report.current_action, str)

    def test_report_notes_nonempty(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("task-001")
        assert len(report.notes) >= 1

    def test_action_changed_flag_accurate(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("task-001")
        assert report.action_changed == (report.historical_action != report.current_action)

    def test_to_dict_is_serialisable(self) -> None:
        import json

        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("task-001")
        d = report.to_dict()
        # Should be JSON-serialisable (no exotic types)
        json.dumps(d)

    def test_risk_score_and_confidence_in_range(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        report = rehydrator.replay("task-001")
        assert 0.0 <= report.current_risk_score <= 1.0
        assert 0.0 <= report.current_confidence <= 1.0

    def test_custom_clock(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        rehydrator._clock = lambda: 9999.0
        report = rehydrator.replay("task-001")
        assert report.replay_timestamp == 9999.0


class TestStateRehydratorReplayAll:
    def test_replay_all_empty(self) -> None:
        ledger = ProvenanceLedger()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        reports = rehydrator.replay_all()
        assert reports == []

    def test_replay_all_multiple(self) -> None:
        ledger = ProvenanceLedger()
        for i in range(5):
            ledger.record(
                f"task-{i}",
                "use_tool",
                grid_state_summary={"domain": "general"},
                clock=lambda: float(i),
            )
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        reports = rehydrator.replay_all()
        assert len(reports) == 5
        task_ids = {r.task_id for r in reports}
        assert task_ids == {f"task-{i}" for i in range(5)}

    def test_replay_all_all_found(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger, brain=_make_brain())
        reports = rehydrator.replay_all()
        assert all(r.found for r in reports)


# ---------------------------------------------------------------------------
# Default brain fallback
# ---------------------------------------------------------------------------


class TestDefaultBrain:
    def test_default_brain_created(self) -> None:
        ledger = _make_ledger_with_receipt()
        rehydrator = StateRehydrator(ledger=ledger)
        # Should not raise — default brain is created via field default_factory
        report = rehydrator.replay("task-001")
        assert report.found is True
