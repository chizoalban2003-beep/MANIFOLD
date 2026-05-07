"""Tests for manifold/trust_network/registry.py"""
from __future__ import annotations
import pytest
from manifold.trust_network.registry import ATSRegistry
from manifold.trust_network.models import ToolRegistration, TrustSignal


def make_reg(tool_id: str = "tool1") -> ToolRegistration:
    return ToolRegistration(tool_id, "org1", "Tool 1", "finance")


def make_signal(
    tool_id: str = "tool1",
    stype: str = "success",
    stakes: float = 0.7,
) -> TrustSignal:
    return TrustSignal(tool_id, stype, "finance", stakes, "org_hash_1")


def test_register_and_score():
    reg = ATSRegistry()
    reg.register_tool(make_reg())
    ats = reg.get_score("tool1")
    assert ats.tier == "provisional"


def test_submit_signals_improves_score():
    reg = ATSRegistry()
    reg.register_tool(make_reg())
    for _ in range(30):
        reg.submit_signal(make_signal())
    ats = reg.get_score("tool1")
    assert ats.tier == "verified"
    assert ats.score > 0.8


def test_adversarial_signals_ban_tool():
    reg = ATSRegistry()
    reg.register_tool(make_reg())
    for _ in range(10):
        reg.submit_signal(make_signal("tool1", "success"))
    for _ in range(5):
        reg.submit_signal(make_signal("tool1", "adversarial", 0.95))
    assert reg.get_score("tool1").tier == "banned"


def test_leaderboard_excludes_banned():
    reg = ATSRegistry()
    reg.register_tool(make_reg("good_tool"))
    reg.register_tool(make_reg("bad_tool"))
    for _ in range(30):
        reg.submit_signal(make_signal("good_tool", "success"))
    for _ in range(5):
        reg.submit_signal(make_signal("bad_tool", "adversarial", 0.95))
    board = reg.leaderboard()
    ids = [s.tool_id for s in board]
    assert "good_tool" in ids
    assert "bad_tool" not in ids


def test_to_dict_returns_expected_keys():
    reg = ATSRegistry()
    reg.register_tool(make_reg())
    d = reg.to_dict("tool1")
    for key in (
        "tool_id", "score", "tier", "total_signals",
        "success_rate", "adversarial_rate", "contributing_orgs",
    ):
        assert key in d


def test_get_all_scores_returns_list():
    reg = ATSRegistry()
    reg.register_tool(make_reg("tool_a"))
    reg.register_tool(make_reg("tool_b"))
    scores = reg.get_all_scores()
    assert len(scores) == 2


def test_submit_without_register_works():
    reg = ATSRegistry()
    # Submit signals for an unregistered tool - should still work
    for _ in range(5):
        reg.submit_signal(make_signal("unknown_tool", "success"))
    ats = reg.get_score("unknown_tool")
    assert ats.total_signals == 5
