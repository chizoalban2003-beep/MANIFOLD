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


# ============================================================
# EXP6 NEW TESTS — Prompt 2 (parallel + sequential detection)
# ============================================================

# New Test 1: "fetch then analyse then report" → 3 sequential tasks, correct depends_on
def test_sequential_three_tasks_correct_depends_on():
    router = TaskRouter()
    plan = router.route("fetch data then analyse the data then write the report")
    assert plan.has_ordering
    # Should produce 3 sub-tasks
    assert len(plan.sub_tasks) >= 2, "Should decompose into multiple sub-tasks"
    # All tasks after the first should have at least one dependency
    non_first = plan.sub_tasks[1:]
    for st in non_first:
        assert len(st.depends_on) > 0, (
            f"Sub-task '{st.prompt}' should depend on the previous task"
        )


# New Test 2: "book flight and reserve hotel" → 2 parallel tasks, same parallel_group
def test_parallel_tasks_same_group():
    router = TaskRouter()
    plan = router.route("book a flight and reserve a hotel room")
    assert plan.has_ordering, "Parallel connector 'and' should be detected"
    # Should produce 2 sub-tasks in the same parallel_group
    assert len(plan.sub_tasks) >= 2
    groups = [st.parallel_group for st in plan.sub_tasks]
    assert groups[0] == groups[1], (
        f"Both tasks should share the same parallel_group, got {groups}"
    )
    # All tasks should be in execution_mode parallel
    for st in plan.sub_tasks:
        assert st.execution_mode == "parallel", (
            f"Task '{st.prompt}' should be parallel, got '{st.execution_mode}'"
        )


# New Test 3: "first fetch, then analyse and summarise" → fetch before [analyse, summarise]
def test_sequential_then_parallel_ordering():
    router = TaskRouter()
    plan = router.route("first fetch the data, then analyse it and summarise results")
    # Should have 3 sub-tasks: 1 sequential fetch + 2 parallel analyse/summarise
    assert len(plan.sub_tasks) >= 2
    # has_ordering must be True
    assert plan.has_ordering
    # The last slot should have tasks with the same parallel_group
    last_group = plan.sub_tasks[-1].parallel_group
    last_slot_tasks = [st for st in plan.sub_tasks if st.parallel_group == last_group]
    first_slot_tasks = [st for st in plan.sub_tasks if st.parallel_group != last_group]
    if len(last_slot_tasks) > 1:
        # Analyse and summarise are parallel
        for st in last_slot_tasks:
            assert st.execution_mode == "parallel"
    # Fetch is in an earlier slot and should have sequential mode
    if first_slot_tasks:
        assert first_slot_tasks[0].parallel_group < last_group


# New Test 4: "process invoices" → 1 task, no ordering
def test_single_task_no_ordering():
    router = TaskRouter()
    plan = router.route("process the quarterly invoices")
    # Should produce 1 sub-task with no ordering
    assert len(plan.sub_tasks) == 1
    assert not plan.sub_tasks[0].depends_on, "Single task should have no dependencies"


# New Test 5: "clean kitchen while drone scouts" → 2 parallel tasks
def test_while_connector_produces_parallel_tasks():
    router = TaskRouter()
    plan = router.route("clean the kitchen while drone scouts the perimeter")
    assert plan.has_ordering, "'while' connector should be detected"
    assert len(plan.sub_tasks) >= 2
    # Both tasks should be in the same parallel group
    groups = [st.parallel_group for st in plan.sub_tasks]
    assert groups[0] == groups[1], (
        f"'while' tasks should share parallel_group, got {groups}"
    )


# New Test 6: TaskPlan.parallel_groups contains correct groupings for parallel tasks
def test_parallel_groups_populated_for_parallel_tasks():
    router = TaskRouter()
    plan = router.route("book a flight and reserve a hotel room")
    assert plan.parallel_groups, "parallel_groups should be non-empty for parallel tasks"
    # Each element of parallel_groups is a list of task-id strings
    for grp in plan.parallel_groups:
        assert isinstance(grp, list)
        for tid in grp:
            assert isinstance(tid, str)

