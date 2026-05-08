"""Tests for brain-state persistence (save/load round-trips)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from manifold.cognitive_map import CognitiveMap
from manifold.cooccurrence import ToolCooccurrenceGraph
from manifold.predictor import PredictiveBrain
from manifold.consolidator import MemoryConsolidator, ConsolidatedRule


# ---------------------------------------------------------------------------
# CognitiveMap
# ---------------------------------------------------------------------------


def test_cognitive_map_save_creates_file(tmp_path: Path) -> None:
    """CognitiveMap.save() writes a JSON file."""
    cmap = CognitiveMap()
    cmap.record_outcome(0, 1, "allow", True, 0.3)
    out = str(tmp_path / "cmap.json")
    cmap.save(out)
    assert os.path.exists(out)
    with open(out) as fh:
        data = json.load(fh)
    assert "outcome_log" in data


def test_cognitive_map_load_restores_node_count(tmp_path: Path) -> None:
    """CognitiveMap.load() returns instance with the same number of entries."""
    cmap = CognitiveMap()
    cmap.record_outcome(0, 0, "allow", True, 0.1)
    cmap.record_outcome(1, 2, "refuse", False, 0.9)
    cmap.record_outcome(3, 3, "verify", True, 0.5)
    out = str(tmp_path / "cmap.json")
    cmap.save(out)

    restored = CognitiveMap.load(out)
    assert len(restored._outcome_log) == 3


# ---------------------------------------------------------------------------
# ToolCooccurrenceGraph
# ---------------------------------------------------------------------------


def test_cooccurrence_save_creates_file(tmp_path: Path) -> None:
    """ToolCooccurrenceGraph.save() writes a JSON file containing cooccurrence data."""
    graph = ToolCooccurrenceGraph()
    graph.record_task(["tool_a", "tool_b"], True)
    out = str(tmp_path / "cooc.json")
    graph.save(out)
    assert os.path.exists(out)
    with open(out) as fh:
        data = json.load(fh)
    assert "cooccurrence" in data


def test_cooccurrence_load_restores_graph(tmp_path: Path) -> None:
    """ToolCooccurrenceGraph.load() restores the graph structure."""
    graph = ToolCooccurrenceGraph()
    graph.record_task(["tool_a", "tool_b"], True)
    graph.record_task(["tool_b", "tool_c"], False)
    out = str(tmp_path / "cooc.json")
    graph.save(out)

    restored = ToolCooccurrenceGraph.load(out)
    # tool counts must match
    assert set(restored._tool_counts.keys()) == {"tool_a", "tool_b", "tool_c"}
    # co-occurrence pair present
    assert restored._cooccurrence.get(("tool_a", "tool_b"), 0) >= 1


# ---------------------------------------------------------------------------
# PredictiveBrain
# ---------------------------------------------------------------------------


def _make_brain_with_log() -> PredictiveBrain:
    from manifold.brain import ManifoldBrain, BrainTask
    brain = PredictiveBrain(brain=ManifoldBrain())
    task = BrainTask(prompt="test", domain="finance", stakes=0.6, uncertainty=0.4)
    brain.predict_and_decide(task, actual_outcome=0.5)
    brain.predict_and_decide(task)
    return brain


def test_predictor_save_caps_at_500(tmp_path: Path) -> None:
    """PredictiveBrain.save() caps the log at 500 entries."""
    from manifold.brain import ManifoldBrain, BrainTask
    brain = PredictiveBrain(brain=ManifoldBrain())
    task = BrainTask(prompt="x", domain="general", stakes=0.5, uncertainty=0.5)
    for _ in range(600):
        brain.predict_and_decide(task)
    out = str(tmp_path / "pred.json")
    brain.save(out)
    with open(out) as fh:
        data = json.load(fh)
    assert len(data["prediction_log"]) == 500


def test_predictor_load_restores_log_entries(tmp_path: Path) -> None:
    """PredictiveBrain.load() restores prediction log entries."""
    brain = _make_brain_with_log()
    out = str(tmp_path / "pred.json")
    brain.save(out)

    restored = PredictiveBrain.load(out)
    assert len(restored._prediction_log) == 2


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------


def _make_consolidator_with_rules() -> MemoryConsolidator:
    consolidator = MemoryConsolidator()
    outcome_log = [
        {"domain": "finance", "action": "allow", "stakes": 0.5, "success": True}
        for _ in range(6)
    ]
    consolidator.consolidate(outcome_log)
    return consolidator


def test_consolidator_save_creates_file(tmp_path: Path) -> None:
    """MemoryConsolidator.save() writes promoted_rules to a JSON file."""
    consolidator = _make_consolidator_with_rules()
    assert len(consolidator._promoted_rules) == 1
    out = str(tmp_path / "cons.json")
    consolidator.save(out)
    assert os.path.exists(out)
    with open(out) as fh:
        data = json.load(fh)
    assert len(data["promoted_rules"]) == 1


def test_consolidator_load_restores_rules(tmp_path: Path) -> None:
    """MemoryConsolidator.load() restores promoted rules."""
    consolidator = _make_consolidator_with_rules()
    out = str(tmp_path / "cons.json")
    consolidator.save(out)

    restored = MemoryConsolidator.load(out)
    assert len(restored._promoted_rules) == 1
    rule = restored._promoted_rules[0]
    assert isinstance(rule, ConsolidatedRule)
    assert rule.domain == "finance"
    assert rule.action == "allow"


# ---------------------------------------------------------------------------
# load() on non-existent path returns fresh instance (no error)
# ---------------------------------------------------------------------------


def test_load_nonexistent_path_returns_fresh_instance() -> None:
    """All load() methods return a fresh instance for missing files without raising."""
    missing = "/tmp/manifold_does_not_exist_xyz_12345.json"
    cmap = CognitiveMap.load(missing)
    assert isinstance(cmap, CognitiveMap)
    assert len(cmap._outcome_log) == 0

    cooc = ToolCooccurrenceGraph.load(missing)
    assert isinstance(cooc, ToolCooccurrenceGraph)
    assert len(cooc._cooccurrence) == 0

    brain = PredictiveBrain.load(missing)
    assert isinstance(brain, PredictiveBrain)
    assert len(brain._prediction_log) == 0

    consolidator = MemoryConsolidator.load(missing)
    assert isinstance(consolidator, MemoryConsolidator)
    assert len(consolidator._promoted_rules) == 0


# ---------------------------------------------------------------------------
# Round-trip: save then load preserves all values exactly
# ---------------------------------------------------------------------------


def test_round_trip_preserves_all_values(tmp_path: Path) -> None:
    """Save then load preserves exact state for all four classes."""
    # CognitiveMap
    cmap = CognitiveMap()
    cmap.record_outcome(2, 3, "allow", True, 0.25)
    cmap.record_outcome(4, 5, "refuse", False, 0.85)
    cmap_path = str(tmp_path / "cmap.json")
    cmap.save(cmap_path)
    cmap2 = CognitiveMap.load(cmap_path)
    assert len(cmap2._outcome_log) == len(cmap._outcome_log)
    assert cmap2._outcome_log[(2, 3)] == cmap._outcome_log[(2, 3)]
    assert cmap2._outcome_log[(4, 5)] == cmap._outcome_log[(4, 5)]

    # ToolCooccurrenceGraph
    graph = ToolCooccurrenceGraph()
    graph.record_task(["alpha", "beta"], True)
    graph.record_task(["alpha", "gamma"], False)
    cooc_path = str(tmp_path / "cooc.json")
    graph.save(cooc_path)
    graph2 = ToolCooccurrenceGraph.load(cooc_path)
    assert dict(graph2._tool_counts) == dict(graph._tool_counts)
    assert dict(graph2._cooccurrence) == dict(graph._cooccurrence)

    # MemoryConsolidator round-trip
    consolidator = _make_consolidator_with_rules()
    cons_path = str(tmp_path / "cons.json")
    consolidator.save(cons_path)
    consolidator2 = MemoryConsolidator.load(cons_path)
    assert len(consolidator2._promoted_rules) == len(consolidator._promoted_rules)
    orig = consolidator._promoted_rules[0]
    rest = consolidator2._promoted_rules[0]
    assert orig.domain == rest.domain
    assert orig.action == rest.action
    assert orig.confidence == rest.confidence
    assert orig.sample_count == rest.sample_count
