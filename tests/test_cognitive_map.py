"""Tests for manifold.cognitive_map."""
import pytest
from manifold.cognitive_map import CognitiveMap
from manifold.brain import ManifoldBrain, BrainTask
from manifold.gridmapper import CellVector


def _brain():
    return ManifoldBrain()


def _task(stakes=0.5, uncertainty=0.5):
    return BrainTask(prompt="test", stakes=stakes, uncertainty=uncertainty)


def test_navigate_returns_k_results():
    brain = _brain()
    task = _task()
    world = brain.map_task_to_world(task)
    cmap = CognitiveMap()
    query = CellVector(cost=0.5, risk=0.5, neutrality=0.5, asset=0.5)
    results = cmap.navigate(query, world, k=3)
    assert len(results) == 3


def test_navigate_sorted_ascending():
    brain = _brain()
    task = _task()
    world = brain.map_task_to_world(task)
    cmap = CognitiveMap()
    query = CellVector(cost=0.5, risk=0.5, neutrality=0.5, asset=0.5)
    results = cmap.navigate(query, world, k=5)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_suggest_action_fallback_when_no_history():
    brain = _brain()
    task = _task()
    world = brain.map_task_to_world(task)
    cmap = CognitiveMap()
    query = CellVector(cost=0.5, risk=0.5, neutrality=0.5, asset=0.5)
    result = cmap.suggest_action(query, world, fallback="verify")
    assert result == "verify"


def test_suggest_action_returns_logged_action():
    brain = _brain()
    task = _task()
    world = brain.map_task_to_world(task)
    cmap = CognitiveMap()
    # Record a successful outcome
    cmap.record_outcome(0, 0, "answer", success=True, risk_score=0.1)
    query = world.cells[0][0]  # exact match → nearest
    result = cmap.suggest_action(query, world, fallback="verify")
    assert result == "answer"


def test_record_outcome_stores_entry():
    cmap = CognitiveMap()
    cmap.record_outcome(2, 3, "use_tool", success=True, risk_score=0.4)
    assert (2, 3) in cmap._outcome_log
    assert cmap._outcome_log[(2, 3)][0]["action"] == "use_tool"


def test_low_vs_high_stakes_risk_scores():
    brain = _brain()
    low_task = BrainTask(prompt="low stakes", stakes=0.1, uncertainty=0.1)
    high_task = BrainTask(prompt="high stakes", stakes=0.9, uncertainty=0.9)
    low_decision = brain.decide(low_task)
    high_decision = brain.decide(high_task)
    assert high_decision.risk_score != low_decision.risk_score
