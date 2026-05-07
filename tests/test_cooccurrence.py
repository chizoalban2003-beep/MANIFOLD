"""Tests for manifold.cooccurrence."""
import pytest
from manifold.cooccurrence import ToolCooccurrenceGraph


def test_single_tool_zero_correlation():
    g = ToolCooccurrenceGraph()
    g.record_task(["tool_a"], True)
    assert g.correlation("tool_a", "tool_b") == 0.0


def test_always_together_correlation_one():
    g = ToolCooccurrenceGraph()
    for _ in range(5):
        g.record_task(["tool_a", "tool_b"], True)
    assert g.correlation("tool_a", "tool_b") == 1.0


def test_partial_cooccurrence():
    g = ToolCooccurrenceGraph()
    # 2 joint, 3 solo-a, 3 solo-b → count_a=5, count_b=5, cooccurrence=2 → 2/5=0.4
    for _ in range(2):
        g.record_task(["tool_a", "tool_b"], True)
    for _ in range(3):
        g.record_task(["tool_a"], True)
    for _ in range(3):
        g.record_task(["tool_b"], True)
    corr = g.correlation("tool_a", "tool_b")
    assert 0.0 < corr < 1.0


def test_propagate_flag_returns_correlated():
    g = ToolCooccurrenceGraph()
    for _ in range(5):
        g.record_task(["tool_a", "tool_b"], True)
    g.min_correlation = 0.0  # ensure we get results
    flagged = g.propagate_flag("tool_a")
    assert "tool_b" in flagged


def test_success_rate_unseen_tool():
    g = ToolCooccurrenceGraph()
    assert g.success_rate("unknown_tool") == 1.0


def test_success_rate_averages():
    g = ToolCooccurrenceGraph()
    g.record_task(["tool_a"], True)
    g.record_task(["tool_a"], True)
    g.record_task(["tool_a"], False)
    assert abs(g.success_rate("tool_a") - (2 / 3)) < 1e-9


def test_summary_contains_all_tools():
    g = ToolCooccurrenceGraph()
    g.record_task(["alpha", "beta"], True)
    g.record_task(["gamma"], False)
    s = g.summary()
    assert "alpha" in s
    assert "beta" in s
    assert "gamma" in s
