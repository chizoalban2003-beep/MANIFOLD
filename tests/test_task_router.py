"""Tests for manifold/task_router.py."""
from __future__ import annotations

import json

import pytest

from manifold.agent_registry import AgentRegistry
from manifold.task_router import SubTask, TaskPlan, TaskRouter


def _router(registry: AgentRegistry | None = None) -> TaskRouter:
    return TaskRouter(registry=registry or AgentRegistry())


def test_route_returns_task_plan():
    router = _router()
    plan = router.route("generate monthly report")
    assert isinstance(plan, TaskPlan)
    assert plan.task_id
    assert plan.original_task == "generate monthly report"
    assert isinstance(plan.sub_tasks, list)
    assert hasattr(plan, "executable")


def test_route_multi_clause_returns_multiple_subtasks():
    router = _router()
    plan = router.route("fetch data and run report and send email")
    assert len(plan.sub_tasks) >= 2


def test_subtask_has_required_fields():
    router = _router()
    plan = router.route("check order status and update inventory")
    for st in plan.sub_tasks:
        assert hasattr(st, "index")
        assert hasattr(st, "prompt")
        assert hasattr(st, "domain")
        assert hasattr(st, "stakes")
        assert hasattr(st, "risk_score")
        assert hasattr(st, "action")
        assert hasattr(st, "status")


def test_high_risk_subtask_is_blocked():
    router = _router()
    plan = router.route("delete all user records permanently now", stakes_hint=0.95)
    blocked = [s for s in plan.sub_tasks if s.status == "blocked"]
    assert len(blocked) > 0


def test_task_plan_to_dict_is_json_serialisable():
    router = _router()
    plan = router.route("generate monthly report")
    d = plan.to_dict()
    # Must not raise
    serialised = json.dumps(d)
    parsed = json.loads(serialised)
    assert parsed["task_id"] == plan.task_id


def test_decompose_splits_connectors():
    router = _router()
    parts = router._decompose("fetch data and run report and send email")
    assert len(parts) >= 3


def test_route_with_no_agents_returns_pending():
    router = _router(registry=AgentRegistry())  # empty registry
    plan = router.route("analyse sales figures")
    # Sub-tasks should be pending (not an error), since governance may permit them
    non_blocked = [s for s in plan.sub_tasks if s.status != "blocked"]
    for st in non_blocked:
        assert st.status == "pending"
        assert st.assigned_to is None


def test_get_plan_retrieves_plan_by_task_id():
    router = _router()
    plan = router.route("check system health")
    retrieved = router.get_plan(plan.task_id)
    assert retrieved is plan
    assert router.get_plan("nonexistent") is None
