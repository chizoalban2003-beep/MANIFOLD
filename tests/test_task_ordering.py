"""Tests for EXP6 — Temporal Task Ordering in TaskRouter."""
from __future__ import annotations

import pytest

from manifold.task_router import DependencyGraph, TaskRouter, run_task_ordering_benchmark


# Test 1: DependencyGraph.topological_sort returns correct order
def test_topological_sort_correct_order():
    graph = DependencyGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    sorted_ids = graph.topological_sort(["a", "b", "c"])
    assert sorted_ids is not None
    assert sorted_ids.index("a") < sorted_ids.index("b")
    assert sorted_ids.index("b") < sorted_ids.index("c")


# Test 2: Cycle detection returns None
def test_topological_sort_cycle_returns_none():
    graph = DependencyGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "a")  # cycle!
    result = graph.topological_sort(["a", "b", "c"])
    assert result is None, f"Expected None for cycle, got {result}"


# Test 3: TaskRouter.route with "then" keyword produces ordered sub-tasks
def test_route_then_keyword_produces_ordering():
    router = TaskRouter()
    plan = router.route("fetch data, then analyse it, then write the report")
    assert plan.has_ordering, "Plan should have has_ordering=True for 'then' keywords"
    # Sub-tasks should have depends_on populated for non-first tasks
    ordered = [st for st in plan.sub_tasks if st.depends_on]
    assert len(ordered) > 0, "Some sub-tasks should have depends_on set"
    # The dependency graph should be non-empty
    assert plan.dependency_graph, "dependency_graph should be non-empty"


# Test 4: parallel_groups has no conflicts within each group
def test_parallel_groups_no_internal_conflicts():
    graph = DependencyGraph()
    graph.add_edge("a", "c")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")

    groups = graph.parallel_groups()
    assert len(groups) > 0

    # Within each group, no two tasks should have a direct edge between them
    all_edges = {(f, t) for f, tos in graph._edges.items() for t in tos}
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ti, tj = group[i], group[j]
                assert (ti, tj) not in all_edges, (
                    f"Conflict within group: {ti} -> {tj}"
                )
                assert (tj, ti) not in all_edges, (
                    f"Conflict within group: {tj} -> {ti}"
                )


# Additional: run_task_ordering_benchmark returns expected keys
def test_task_ordering_benchmark_returns_keys():
    result = run_task_ordering_benchmark()
    assert isinstance(result, dict)
    assert "has_ordering" in result
    assert result["has_ordering"] is True
    assert "dependency_count" in result
    assert result["dependency_count"] > 0
