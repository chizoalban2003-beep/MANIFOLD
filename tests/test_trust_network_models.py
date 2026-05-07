"""Tests for manifold/trust_network/models.py"""
from __future__ import annotations
import pytest
from manifold.trust_network.models import TrustSignal, AgentTrustScore, ToolRegistration


def test_ats_from_empty_signals():
    ats = AgentTrustScore.from_signals("tool_x", [])
    assert ats.tier == "provisional"
    assert ats.score == 0.5


def test_ats_verified_after_many_successes():
    signals = [
        TrustSignal("tool_x", "success", "finance", 0.7, "org1")
        for _ in range(30)
    ]
    ats = AgentTrustScore.from_signals("tool_x", signals)
    assert ats.tier == "verified"
    assert ats.score > 0.8


def test_ats_banned_after_adversarial_signals():
    signals = (
        [TrustSignal("tool_x", "success", "finance", 0.5, "org1")] * 10
        + [TrustSignal("tool_x", "adversarial", "finance", 0.9, "org2")] * 5
    )
    ats = AgentTrustScore.from_signals("tool_x", signals)
    assert ats.tier == "banned"


def test_ats_provisional_under_20_signals():
    signals = [TrustSignal("tool_x", "success", "general", 0.5, "org1")] * 10
    ats = AgentTrustScore.from_signals("tool_x", signals)
    assert ats.tier == "provisional"


def test_contributing_orgs_counted():
    signals = [
        TrustSignal("tool_x", "success", "general", 0.5, "org_a"),
        TrustSignal("tool_x", "success", "general", 0.5, "org_b"),
        TrustSignal("tool_x", "success", "general", 0.5, "org_a"),
    ]
    ats = AgentTrustScore.from_signals("tool_x", signals)
    assert ats.contributing_orgs == 2


def test_tool_registration_fields():
    reg = ToolRegistration("api-v1", "acme", "Acme Billing API", "finance")
    assert reg.tool_id == "api-v1"
    assert reg.domain == "finance"


def test_ats_flagged_low_success_rate():
    signals = (
        [TrustSignal("tool_x", "success", "finance", 0.5, "org1")] * 8
        + [TrustSignal("tool_x", "failure", "finance", 0.5, "org1")] * 12
    )
    ats = AgentTrustScore.from_signals("tool_x", signals)
    assert ats.tier in ("flagged", "provisional")
