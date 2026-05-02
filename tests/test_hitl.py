"""Tests for Phase 9: HITLGate, HITLConfig, HITLRecord, TeacherSpike."""

import pytest

from manifold import (
    BrainConfig,
    BrainMemory,
    BrainTask,
    HITLConfig,
    HITLGate,
    HITLRecord,
    ManifoldBrain,
    TeacherSpike,
    ToolProfile,
    default_tools,
)


_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)
_TOOLS = default_tools()


# ---------------------------------------------------------------------------
# HITLConfig defaults
# ---------------------------------------------------------------------------


def test_hitl_config_defaults() -> None:
    cfg = HITLConfig()
    assert cfg.risk_stakes_threshold == pytest.approx(0.55)
    assert "refuse" in cfg.force_escalate_actions
    assert cfg.human_cost == pytest.approx(0.45)
    assert cfg.teacher_spike_weight == pytest.approx(0.85)


def test_hitl_config_custom() -> None:
    cfg = HITLConfig(risk_stakes_threshold=0.70, teacher_spike_weight=0.60)
    assert cfg.risk_stakes_threshold == pytest.approx(0.70)
    assert cfg.teacher_spike_weight == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# TeacherSpike.apply_to_memory
# ---------------------------------------------------------------------------


def test_teacher_spike_approve_raises_success_rate() -> None:
    memory = BrainMemory()
    memory.tool_stats["legal_search"] = {
        "success_rate": 0.40, "count": 10.0, "utility": 0.6, "consecutive_failures": 2.0
    }
    spike = TeacherSpike(tool_name="legal_search", verdict="approve", confidence=0.90)
    spike.apply_to_memory(memory, spike_weight=0.85)
    assert memory.tool_stats["legal_search"]["success_rate"] > 0.40


def test_teacher_spike_reject_lowers_success_rate() -> None:
    memory = BrainMemory()
    memory.tool_stats["bad_tool"] = {
        "success_rate": 0.90, "count": 5.0, "utility": 0.7, "consecutive_failures": 0.0
    }
    spike = TeacherSpike(tool_name="bad_tool", verdict="reject", confidence=0.90)
    spike.apply_to_memory(memory, spike_weight=0.85)
    assert memory.tool_stats["bad_tool"]["success_rate"] < 0.90


def test_teacher_spike_correct_lowers_success_rate() -> None:
    memory = BrainMemory()
    memory.tool_stats["tool_x"] = {
        "success_rate": 0.80, "count": 5.0, "utility": 0.6, "consecutive_failures": 0.0
    }
    spike = TeacherSpike(tool_name="tool_x", verdict="correct", confidence=0.95)
    spike.apply_to_memory(memory, spike_weight=0.85)
    assert memory.tool_stats["tool_x"]["success_rate"] < 0.80


def test_teacher_spike_creates_entry_if_missing() -> None:
    memory = BrainMemory()
    spike = TeacherSpike(tool_name="new_tool", verdict="approve")
    spike.apply_to_memory(memory)
    assert "new_tool" in memory.tool_stats


def test_teacher_spike_stronger_than_normal_learning() -> None:
    """Super-scar (0.85 weight) should produce a larger delta than normal LR (0.15)."""
    mem_normal = BrainMemory(base_learning_rate=0.15)
    mem_super = BrainMemory(base_learning_rate=0.15)
    for mem in (mem_normal, mem_super):
        mem.tool_stats["tool_y"] = {
            "success_rate": 0.80, "count": 5.0, "utility": 0.6, "consecutive_failures": 0.0
        }
    # Normal learning rate update (simulate)
    mem_normal.tool_stats["tool_y"]["success_rate"] = (
        0.80 * (1 - 0.15) + 0.0 * 0.15  # reject with normal LR
    )
    # Super-scar
    spike = TeacherSpike(tool_name="tool_y", verdict="reject", confidence=1.0)
    spike.apply_to_memory(mem_super, spike_weight=0.85)

    normal_drop = 0.80 - mem_normal.tool_stats["tool_y"]["success_rate"]
    super_drop = 0.80 - mem_super.tool_stats["tool_y"]["success_rate"]
    assert super_drop > normal_drop


def test_teacher_spike_increments_count() -> None:
    memory = BrainMemory()
    memory.tool_stats["t"] = {
        "success_rate": 0.50, "count": 5.0, "utility": 0.5, "consecutive_failures": 0.0
    }
    spike = TeacherSpike(tool_name="t", verdict="approve")
    spike.apply_to_memory(memory)
    assert memory.tool_stats["t"]["count"] == 6.0


def test_teacher_spike_reject_increments_consecutive_failures() -> None:
    memory = BrainMemory()
    memory.tool_stats["t"] = {
        "success_rate": 0.80, "count": 5.0, "utility": 0.5, "consecutive_failures": 0.0
    }
    spike = TeacherSpike(tool_name="t", verdict="reject")
    spike.apply_to_memory(memory)
    assert memory.tool_stats["t"]["consecutive_failures"] > 0.0


# ---------------------------------------------------------------------------
# HITLGate.should_escalate
# ---------------------------------------------------------------------------


def test_hitl_gate_escalates_high_risk_stakes() -> None:
    gate = HITLGate(config=HITLConfig(risk_stakes_threshold=0.50))
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Process medical record", "medical", stakes=0.9, safety_sensitivity=0.9)
    decision = brain.decide(task)
    # Manually set risk_score to force escalation
    import dataclasses as _dc
    high_risk_decision = _dc.replace(decision, risk_score=0.65)
    assert gate.should_escalate(task, high_risk_decision)


def test_hitl_gate_no_escalation_low_risk() -> None:
    gate = HITLGate(config=HITLConfig(risk_stakes_threshold=0.80))
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("What is 2+2?", "math", stakes=0.10)
    decision = brain.decide(task)
    import dataclasses as _dc
    low_risk = _dc.replace(decision, risk_score=0.10, action="answer")
    assert not gate.should_escalate(task, low_risk)


def test_hitl_gate_force_escalate_on_refuse() -> None:
    gate = HITLGate(config=HITLConfig(risk_stakes_threshold=0.99))  # very high threshold
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Dangerous task", "general")
    decision = brain.decide(task)
    refuse_decision = _dc.replace(decision, action="refuse", risk_score=0.10)
    assert gate.should_escalate(task, refuse_decision)


# ---------------------------------------------------------------------------
# HITLGate.escalate + resolve
# ---------------------------------------------------------------------------


def test_hitl_gate_escalate_queues_record() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Dangerous", "general", stakes=0.9)
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    record = gate.escalate(task, decision)
    assert isinstance(record, HITLRecord)
    assert not record.resolved
    assert len(gate.pending_queue()) == 1


def test_hitl_gate_resolve_clears_pending() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Dangerous", "general", stakes=0.9)
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    record = gate.escalate(task, decision)
    gate.resolve(record, verdict="reject")
    assert len(gate.pending_queue()) == 0
    assert len(gate.resolved_records()) == 1


def test_hitl_gate_resolve_applies_teacher_spike() -> None:
    memory = BrainMemory()
    memory.tool_stats["general"] = {
        "success_rate": 0.90, "count": 5.0, "utility": 0.6, "consecutive_failures": 0.0
    }
    gate = HITLGate(memory=memory)
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Risky legal action", "general", stakes=0.9)
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    record = gate.escalate(task, decision)
    gate.resolve(record, verdict="reject", tool_name="general")
    # Success rate should have dropped due to super-scar
    assert memory.tool_stats["general"]["success_rate"] < 0.90


def test_hitl_gate_resolve_returns_teacher_spike() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Task", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    record = gate.escalate(task, decision)
    spike = gate.resolve(record, verdict="approve")
    assert isinstance(spike, TeacherSpike)
    assert spike.verdict == "approve"


def test_hitl_gate_spike_history_grows() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("Task", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    for _ in range(3):
        rec = gate.escalate(task, decision)
        gate.resolve(rec, verdict="approve")
    assert len(gate.spike_history()) == 3


def test_hitl_gate_escalation_count() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("T", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    gate.escalate(task, decision)
    gate.escalate(task, decision)
    assert gate.escalation_count() == 2


def test_hitl_gate_escalation_cost() -> None:
    cfg = HITLConfig(human_cost=0.30)
    gate = HITLGate(config=cfg)
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("T", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    gate.escalate(task, decision)
    gate.escalate(task, decision)
    assert gate.escalation_cost() == pytest.approx(0.60)


def test_hitl_gate_rejection_rate() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("T", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    r1 = gate.escalate(task, decision)
    r2 = gate.escalate(task, decision)
    gate.resolve(r1, verdict="approve")
    gate.resolve(r2, verdict="reject")
    assert gate.rejection_rate() == pytest.approx(0.5)


def test_hitl_gate_rejection_rate_empty_returns_zero() -> None:
    gate = HITLGate()
    assert gate.rejection_rate() == 0.0


def test_hitl_gate_to_brain_outcome_approve() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("T", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    record = gate.escalate(task, decision)
    gate.resolve(record, verdict="approve")
    outcome = gate.to_brain_outcome(record)
    assert outcome.success
    assert outcome.cost_paid == pytest.approx(gate.config.human_cost)


def test_hitl_gate_to_brain_outcome_reject() -> None:
    gate = HITLGate()
    import dataclasses as _dc
    brain = ManifoldBrain(_CFG, tools=_TOOLS)
    task = BrainTask("T", "general")
    decision = _dc.replace(brain.decide(task), risk_score=0.70)
    record = gate.escalate(task, decision)
    gate.resolve(record, verdict="reject")
    outcome = gate.to_brain_outcome(record)
    assert not outcome.success
    assert outcome.risk_realized > 0.0


# ---------------------------------------------------------------------------
# Integration: HITLGate + ManifoldBrain
# ---------------------------------------------------------------------------


def test_hitl_gate_full_loop_brain_learn() -> None:
    """Full loop: brain decides, gate escalates, human resolves, brain learns."""
    memory = BrainMemory()
    brain = ManifoldBrain(_CFG, tools=_TOOLS, memory=memory)
    gate = HITLGate(config=HITLConfig(risk_stakes_threshold=0.40), memory=memory)

    import dataclasses as _dc
    task = BrainTask("Dangerous legal action", "legal", stakes=0.9, safety_sensitivity=0.8)
    decision = brain.decide(task)
    high_risk = _dc.replace(decision, risk_score=0.65)

    if gate.should_escalate(task, high_risk):
        record = gate.escalate(task, high_risk)
        gate.resolve(record, verdict="reject", tool_name="legal")

    # Memory should now have a scar for "legal" domain
    assert "legal" in memory.tool_stats
    assert memory.tool_stats["legal"]["success_rate"] < 1.0
