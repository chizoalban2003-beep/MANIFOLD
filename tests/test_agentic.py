"""End-to-end tests for the MANIFOLD agentic layer.

Tests: AgentRegistry, AgentMonitor, TaskRouter (all three together).
"""
from __future__ import annotations

import json
import time

import pytest

from manifold.agent_registry import AgentRegistry
from manifold.monitor import AgentMonitor
from manifold.task_router import TaskRouter


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------


def test_agent_registers_and_is_found():
    registry = AgentRegistry()
    registry.register(
        "billing-agent",
        "Billing Agent",
        ["billing", "invoice"],
        "acme",
        "",
        domain="finance",
    )
    rec = registry.get("billing-agent")
    assert rec is not None
    assert rec.display_name == "Billing Agent"


def test_heartbeat_prevents_stale():
    registry = AgentRegistry(stale_timeout=120)
    registry.register("hb-agent", "HB Agent", [], "org1")
    time.sleep(0)
    registry.heartbeat("hb-agent")
    rec = registry.get("hb-agent")
    assert rec.is_stale(timeout_seconds=120) is False


def test_pause_and_resume():
    registry = AgentRegistry()
    registry.register("pr-agent", "PR Agent", [], "org1")
    registry.pause("pr-agent")
    assert registry.get("pr-agent").status == "paused"
    registry.resume("pr-agent")
    assert registry.get("pr-agent").status == "active"


def test_stale_detection():
    registry = AgentRegistry(stale_timeout=0)
    registry.register("stale-agent", "Stale Agent", [], "org1")
    time.sleep(0.01)
    registry.mark_stale_agents()
    assert registry.get("stale-agent").status == "stale"


# ---------------------------------------------------------------------------
# TaskRouter tests
# ---------------------------------------------------------------------------


def test_task_router_decomposes_multi_clause():
    router = TaskRouter()
    plan = router.route("fetch the data and run analysis and send report")
    assert len(plan.sub_tasks) >= 2


def test_task_router_blocks_high_risk():
    router = TaskRouter()
    plan = router.route("delete all user records permanently now", stakes_hint=0.95)
    blocked = [s for s in plan.sub_tasks if s.status == "blocked"]
    assert len(blocked) > 0


def test_task_router_plan_serialisable():
    router = TaskRouter()
    plan = router.route("generate monthly report")
    d = plan.to_dict()
    json.dumps(d)  # Must not raise


def test_task_plan_has_required_fields():
    router = TaskRouter()
    plan = router.route("check order status")
    d = plan.to_dict()
    for key in ("task_id", "original_task", "sub_tasks", "executable", "blocked_count"):
        assert key in d


# ---------------------------------------------------------------------------
# AgentMonitor tests
# ---------------------------------------------------------------------------


def test_agent_monitor_status():
    registry = AgentRegistry()
    monitor = AgentMonitor(registry, check_interval=999)
    s = monitor.status()
    assert "running" in s
    assert "registry_summary" in s
    assert "check_interval" in s
    assert "health_threshold" in s


# ---------------------------------------------------------------------------
# Capability filter test
# ---------------------------------------------------------------------------


def test_agents_with_capability_filter():
    registry = AgentRegistry()
    registry.register("a1", "Agent 1", ["billing", "email"], "org1")
    registry.register("a2", "Agent 2", ["search", "code"], "org1")
    billing = registry.agents_with_capability("billing")
    assert len(billing) == 1
    assert billing[0].agent_id == "a1"
