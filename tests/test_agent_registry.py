"""Tests for manifold/agent_registry.py."""
from __future__ import annotations

import time

import pytest

from manifold.agent_registry import AgentRegistry, AgentRecord


def _reg(stale_timeout: int = 120) -> AgentRegistry:
    return AgentRegistry(stale_timeout=stale_timeout)


def test_register_creates_agent_record():
    reg = _reg()
    rec = reg.register(
        agent_id="a1",
        display_name="Test Agent",
        capabilities=["code", "search"],
        org_id="org1",
        endpoint_url="http://localhost:9000",
        domain="devops",
        notes="test note",
    )
    assert isinstance(rec, AgentRecord)
    assert rec.agent_id == "a1"
    assert rec.display_name == "Test Agent"
    assert rec.capabilities == ["code", "search"]
    assert rec.org_id == "org1"
    assert rec.domain == "devops"
    assert rec.status == "active"


def test_heartbeat_updates_timestamp_and_returns_true():
    reg = _reg()
    reg.register("a2", "Agent 2", ["billing"], "org1")
    before = reg.get("a2").last_heartbeat
    time.sleep(0.02)
    result = reg.heartbeat("a2")
    assert result is True
    assert reg.get("a2").last_heartbeat > before


def test_heartbeat_returns_false_for_unknown_agent():
    reg = _reg()
    result = reg.heartbeat("nonexistent")
    assert result is False


def test_pause_sets_status_to_paused():
    reg = _reg()
    reg.register("a3", "Agent 3", [], "org1")
    reg.pause("a3")
    assert reg.get("a3").status == "paused"


def test_resume_sets_status_to_active():
    reg = _reg()
    reg.register("a4", "Agent 4", [], "org1")
    reg.pause("a4")
    reg.resume("a4")
    assert reg.get("a4").status == "active"


def test_agents_with_capability_returns_matching_agents():
    reg = _reg()
    reg.register("a5", "Billing Agent", ["billing", "invoice"], "org1")
    reg.register("a6", "Code Agent", ["code", "search"], "org1")
    reg.register("a7", "Multi Agent", ["billing", "code"], "org1")

    billing_agents = reg.agents_with_capability("billing")
    ids = {a.agent_id for a in billing_agents}
    assert "a5" in ids
    assert "a7" in ids
    assert "a6" not in ids


def test_mark_stale_agents_marks_timed_out_agents():
    reg = AgentRegistry(stale_timeout=0)
    reg.register("stale1", "Stale Agent", [], "org1")
    time.sleep(0.01)
    stale = reg.mark_stale_agents()
    assert "stale1" in stale
    assert reg.get("stale1").status == "stale"


def test_summary_returns_correct_structure():
    reg = _reg()
    reg.register("s1", "Active Agent", [], "org1")
    reg.register("s2", "Paused Agent", [], "org1")
    reg.pause("s2")

    summary = reg.summary()
    assert "total" in summary
    assert "active" in summary
    assert "paused" in summary
    assert "stale" in summary
    assert "avg_health" in summary
    assert summary["total"] == 2
    assert summary["paused"] == 1
    assert isinstance(summary["avg_health"], float)
