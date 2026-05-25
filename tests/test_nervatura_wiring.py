"""Tests for NERVATURA agent wiring — Addition 1, 2, and 3.

Tests
-----
1. AgentCRNAProfile exists with correct fields for "physical/floor"
2. AgentRegistry.register() with layer="physical/floor" sets crna_profile
3. TaskRouter completing a sub-task with a known agent fires CellUpdateBus
4. NERVATURAWorld cell.n decreases after a scout agent task completes
5. NERVATURAWorld cell.c decreases after a builder agent task completes
6. GET /nervatura/zone-crna returns dict with kitchen/devops/finance/legal keys
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from manifold.agent_registry import AgentRegistry, NERVATURA_PROFILES, AgentCRNAProfile
from manifold.cell_update_bus import get_bus, CellUpdateBus
from manifold.nervatura_world import NERVATURAWorld, set_world, get_world
from manifold.task_router import TaskRouter


# ---------------------------------------------------------------------------
# Test 1 — AgentCRNAProfile fields for "physical/floor"
# ---------------------------------------------------------------------------

def test_agent_crna_profile_physical_floor_fields():
    profile = NERVATURA_PROFILES["physical/floor"]
    assert isinstance(profile, AgentCRNAProfile)
    assert profile.archetype == "builder"
    assert profile.c_delta == pytest.approx(-0.12)
    assert profile.r_delta == pytest.approx(-0.08)
    assert profile.n_delta == pytest.approx(0.0)
    assert profile.a_delta == pytest.approx(0.0)
    assert "kitchen" in profile.effective_zones
    assert "physical" in profile.effective_zones


# ---------------------------------------------------------------------------
# Test 2 — AgentRegistry.register() with layer sets crna_profile
# ---------------------------------------------------------------------------

def test_agent_registry_register_sets_crna_profile():
    registry = AgentRegistry(stale_timeout=9999)
    agent = registry.register(
        agent_id="robot-001",
        display_name="Floor Robot",
        capabilities=["navigate", "clean"],
        org_id="org1",
        layer="physical/floor",
    )
    assert agent.layer == "physical/floor"
    assert isinstance(agent.crna_profile, AgentCRNAProfile)
    assert agent.crna_profile.archetype == "builder"
    assert agent.crna_profile.c_delta == pytest.approx(-0.12)


# ---------------------------------------------------------------------------
# Test 3 — TaskRouter.complete_sub_task() fires CellUpdateBus
# ---------------------------------------------------------------------------

def test_complete_sub_task_fires_cell_update_bus():
    # Use a fresh bus so we can inspect what was published
    fresh_bus = CellUpdateBus()
    received = []
    fresh_bus.subscribe("test", lambda u: received.append(u))

    import manifold.task_router as tr_mod
    import manifold.cell_update_bus as bus_mod

    # Patch get_bus in task_router's runtime imports
    original_get_bus = bus_mod.get_bus
    bus_mod.get_bus = lambda: fresh_bus
    try:
        registry = AgentRegistry(stale_timeout=9999)
        registry.register(
            agent_id="scout-001",
            display_name="Scout",
            capabilities=["aerial", "scan"],
            org_id="org1",
            layer="physical/aerial",
        )

        router = TaskRouter(registry=registry)
        plan = router.route("survey the area")

        # Find first assigned sub-task
        assigned = [st for st in plan.sub_tasks if st.assigned_to == "scout-001"]
        if not assigned:
            # Force assign the first sub-task to our agent
            plan.sub_tasks[0].assigned_to = "scout-001"
            plan.sub_tasks[0].status = "assigned"

        st = plan.sub_tasks[0]
        result = router.complete_sub_task(plan.task_id, st.index)
        assert result is True
        assert st.status == "complete"

        # Wait for bus background threads to deliver (with timeout)
        import time
        deadline = time.time() + 2.0
        while time.time() < deadline and len(received) == 0:
            time.sleep(0.02)

        assert st.has_nervatura_effect is True
        assert plan.nervatura_effects_fired >= 1
        # At least one CellUpdate should have been published
        assert len(received) >= 1
    finally:
        bus_mod.get_bus = original_get_bus


# ---------------------------------------------------------------------------
# Test 4 — NERVATURAWorld cell.n decreases after scout agent task completes
# ---------------------------------------------------------------------------

def test_scout_task_complete_reduces_cell_n():
    world = NERVATURAWorld(width=5, depth=5, height=1, default_crna=(0.5, 0.5, 1.0, 0.0))
    set_world(world)
    try:
        registry = AgentRegistry(stale_timeout=9999)
        agent = registry.register(
            agent_id="scout-002",
            display_name="Scout Drone",
            capabilities=["scan", "aerial"],
            org_id="org1",
            layer="physical/aerial",
        )
        # Place the agent at (2, 2, 0)
        agent.position = (2, 2, 0)

        router = TaskRouter(registry=registry)
        plan = router.route("map the terrain")
        # Force assign first sub-task
        plan.sub_tasks[0].assigned_to = "scout-002"
        plan.sub_tasks[0].status = "assigned"

        n_before = world.cell(2, 2, 0).n

        router.complete_sub_task(plan.task_id, plan.sub_tasks[0].index)

        n_after = world.cell(2, 2, 0).n
        # Scout profile has n_delta=-0.22, so n should decrease
        assert n_after < n_before
    finally:
        set_world(None)


# ---------------------------------------------------------------------------
# Test 5 — NERVATURAWorld cell.c decreases after builder agent task completes
# ---------------------------------------------------------------------------

def test_builder_task_complete_reduces_cell_c():
    world = NERVATURAWorld(width=5, depth=5, height=1, default_crna=(0.5, 0.5, 1.0, 0.0))
    set_world(world)
    try:
        registry = AgentRegistry(stale_timeout=9999)
        agent = registry.register(
            agent_id="builder-001",
            display_name="Floor Robot",
            capabilities=["clean", "navigate"],
            org_id="org1",
            layer="physical/floor",
        )
        agent.position = (1, 1, 0)

        router = TaskRouter(registry=registry)
        plan = router.route("clean the kitchen floor")
        plan.sub_tasks[0].assigned_to = "builder-001"
        plan.sub_tasks[0].status = "assigned"

        c_before = world.cell(1, 1, 0).c

        router.complete_sub_task(plan.task_id, plan.sub_tasks[0].index)

        c_after = world.cell(1, 1, 0).c
        # Builder profile has c_delta=-0.12 (terraform), so c should decrease
        assert c_after < c_before
    finally:
        set_world(None)


# ---------------------------------------------------------------------------
# Test 6 — GET /nervatura/zone-crna returns dict with required zone keys
# ---------------------------------------------------------------------------

def test_nervatura_zone_crna_endpoint_returns_zone_keys():
    response = {}

    def fake_send_json(h, status, body):
        response["status"] = status
        response["body"] = body

    import manifold.server as srv
    import manifold.routes.physical as phys_mod

    original_nervatura = srv._NERVATURA
    original_send = srv._send_json

    # Set up a real NERVATURAWorld with enough cells
    world = NERVATURAWorld(width=12, depth=12, height=1)
    srv._NERVATURA = world
    srv._send_json = fake_send_json

    handler = MagicMock()

    try:
        phys_mod.handle_get_nervatura_zone_crna(handler)
        assert response["status"] == 200
        body = response["body"]
        for zone in ("kitchen", "devops", "finance", "legal"):
            assert zone in body, f"Missing zone: {zone}"
            zone_data = body[zone]
            assert "c" in zone_data
            assert "r" in zone_data
            assert "n" in zone_data
            assert "a" in zone_data
        assert "timestamp" in body
    finally:
        srv._NERVATURA = original_nervatura
        srv._send_json = original_send
