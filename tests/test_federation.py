"""Tests for Phase 10: Federated Gossip & Global Reputation Ledger."""

import pytest

from manifold.federation import (
    FederatedGossipBridge,
    FederatedGossipPacket,
    GlobalReputationLedger,
    OrgReputationSnapshot,
    cold_start_from_ledger,
)
from manifold.brain import BrainMemory, ToolProfile
from manifold.transfer import ReputationRegistry


# ---------------------------------------------------------------------------
# FederatedGossipPacket
# ---------------------------------------------------------------------------


def test_packet_healthy_full_confidence():
    p = FederatedGossipPacket("web_search", "healthy", confidence=1.0)
    assert p.implied_success_rate == 1.0


def test_packet_failing_full_confidence():
    p = FederatedGossipPacket("web_search", "failing", confidence=1.0)
    assert p.implied_success_rate == 0.0


def test_packet_degraded_full_confidence():
    p = FederatedGossipPacket("web_search", "degraded", confidence=1.0)
    assert abs(p.implied_success_rate - 0.5) < 1e-6


def test_packet_healthy_half_confidence():
    p = FederatedGossipPacket("web_search", "healthy", confidence=0.5)
    # base=1.0 * 0.5 + (1-0.5) * 0.5 = 0.75
    assert abs(p.implied_success_rate - 0.75) < 1e-6


def test_packet_failing_half_confidence():
    p = FederatedGossipPacket("web_search", "failing", confidence=0.5)
    # base=0.0 * 0.5 + 0.5 * 0.5 = 0.25
    assert abs(p.implied_success_rate - 0.25) < 1e-6


def test_packet_defaults():
    p = FederatedGossipPacket("tool", "healthy")
    assert p.confidence == 1.0
    assert p.org_id == "anonymous"
    assert p.weight == 1.0


# ---------------------------------------------------------------------------
# OrgReputationSnapshot
# ---------------------------------------------------------------------------


def test_snapshot_from_registry_basic():
    reg = ReputationRegistry()
    reg.observe("tool_a", success_rate=0.7, n_observations=10)
    reg.observe("tool_b", success_rate=0.9, n_observations=5)

    snap = OrgReputationSnapshot.from_registry("org_x", reg)
    assert snap.org_id == "org_x"
    assert "tool_a" in snap.rates
    assert "tool_b" in snap.rates
    assert abs(snap.rates["tool_a"][0] - 0.7) < 0.01


def test_snapshot_records_observation_count():
    reg = ReputationRegistry()
    reg.observe("tool_a", success_rate=0.8, n_observations=20)
    snap = OrgReputationSnapshot.from_registry("org_y", reg)
    assert snap.rates["tool_a"][1] == 20


def test_snapshot_empty_registry():
    reg = ReputationRegistry()
    snap = OrgReputationSnapshot.from_registry("org_z", reg)
    assert snap.rates == {}


# ---------------------------------------------------------------------------
# GlobalReputationLedger
# ---------------------------------------------------------------------------


def test_ledger_ingest_snapshot_basic():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    snap = OrgReputationSnapshot("org_a", {"web_search": (0.8, 10)})
    ledger.ingest_snapshot(snap)
    rate = ledger.global_rate("web_search")
    assert rate is not None
    assert abs(rate - 0.8) < 0.05


def test_ledger_merge_two_orgs():
    ledger = GlobalReputationLedger(min_orgs_required=2)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.6, 10)}))
    ledger.ingest_snapshot(OrgReputationSnapshot("org_b", {"tool": (0.8, 10)}))
    rate = ledger.global_rate("tool")
    assert rate is not None
    # weighted average with equal n: (0.6*10 + 0.8*10)/20 = 0.7
    assert abs(rate - 0.7) < 0.05


def test_ledger_is_trustworthy_insufficient_orgs():
    ledger = GlobalReputationLedger(min_orgs_required=2)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.8, 5)}))
    assert not ledger.is_trustworthy("tool")


def test_ledger_is_trustworthy_sufficient_orgs():
    ledger = GlobalReputationLedger(min_orgs_required=2)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.8, 5)}))
    ledger.ingest_snapshot(OrgReputationSnapshot("org_b", {"tool": (0.7, 5)}))
    assert ledger.is_trustworthy("tool")


def test_ledger_unknown_tool_returns_none():
    ledger = GlobalReputationLedger()
    assert ledger.global_rate("nonexistent") is None


def test_ledger_ingest_packet():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    p = FederatedGossipPacket("gpt-4o", "failing", org_id="org_a")
    ledger.ingest_packet(p)
    rate = ledger.global_rate("gpt-4o")
    assert rate is not None
    assert rate < 0.5


def test_ledger_ingest_packet_healthy():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    for _ in range(5):
        ledger.ingest_packet(FederatedGossipPacket("api", "healthy", org_id="org_a"))
    rate = ledger.global_rate("api")
    assert rate is not None
    assert rate > 0.9


def test_ledger_all_rates():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"t1": (0.7, 5), "t2": (0.9, 5)}))
    rates = ledger.all_rates()
    assert "t1" in rates
    assert "t2" in rates


def test_ledger_contributing_org_count():
    ledger = GlobalReputationLedger()
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.8, 5)}))
    ledger.ingest_snapshot(OrgReputationSnapshot("org_b", {"tool": (0.7, 5)}))
    assert ledger.contributing_org_count("tool") == 2


def test_ledger_known_tools():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"x": (0.5, 3), "y": (0.9, 7)}))
    assert set(ledger.known_tools()) == {"x", "y"}


# ---------------------------------------------------------------------------
# FederatedGossipBridge
# ---------------------------------------------------------------------------


def test_bridge_register():
    bridge = FederatedGossipBridge()
    bridge.register("org_a")
    assert "org_a" in bridge.registered_orgs()


def test_bridge_contribute_snapshot_updates_ledger():
    bridge = FederatedGossipBridge()
    snap = OrgReputationSnapshot("org_a", {"web_search": (0.75, 10)})
    bridge.contribute_snapshot(snap)
    assert bridge.ledger.global_rate("web_search") is not None


def test_bridge_contribute_packet_updates_ledger():
    bridge = FederatedGossipBridge()
    p = FederatedGossipPacket("tool_x", "failing", org_id="org_b")
    bridge.contribute_packet(p)
    assert bridge.ledger.global_rate("tool_x") is not None


def test_bridge_global_channel_filter():
    bridge = FederatedGossipBridge(global_channel_tools=frozenset({"shared_tool"}))
    snap = OrgReputationSnapshot("org_a", {
        "shared_tool": (0.8, 5),
        "private_tool": (0.6, 5),
    })
    bridge.contribute_snapshot(snap)
    # Only shared_tool should reach the global ledger
    assert bridge.ledger.global_rate("shared_tool") is not None
    assert bridge.ledger.global_rate("private_tool") is None


def test_bridge_export_snapshot():
    bridge = FederatedGossipBridge()
    reg = ReputationRegistry()
    reg.observe("tool_a", 0.8, 10)
    snap_in = OrgReputationSnapshot.from_registry("org_a", reg)
    bridge.contribute_snapshot(snap_in)
    snap_out = bridge.export_snapshot("org_a")
    assert snap_out is not None
    assert "tool_a" in snap_out.rates


def test_bridge_export_unknown_org_returns_none():
    bridge = FederatedGossipBridge()
    assert bridge.export_snapshot("nonexistent") is None


def test_bridge_org_registry():
    bridge = FederatedGossipBridge()
    bridge.register("org_a")
    reg = bridge.org_registry("org_a")
    assert reg is not None
    assert bridge.org_registry("unknown") is None


# ---------------------------------------------------------------------------
# cold_start_from_ledger
# ---------------------------------------------------------------------------


def test_cold_start_basic():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"web_search": (0.6, 10)}))
    tools = [ToolProfile(name="web_search", cost=0.05, latency=0.1, risk=0.1, asset=0.7, reliability=0.9)]
    memory = cold_start_from_ledger(ledger, tools=tools, alpha=0.5)
    assert "web_search" in memory.tool_stats
    # alpha=0.5: 0.5*0.6 + 0.5*0.9 = 0.75
    sr = memory.tool_stats["web_search"]["success_rate"]
    assert abs(sr - 0.75) < 0.05


def test_cold_start_only_trustworthy():
    ledger = GlobalReputationLedger(min_orgs_required=2)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.6, 5)}))
    # Only one org — not trustworthy
    memory = cold_start_from_ledger(ledger, only_trustworthy=True)
    assert "tool" not in memory.tool_stats


def test_cold_start_not_only_trustworthy():
    ledger = GlobalReputationLedger(min_orgs_required=2)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.6, 5)}))
    memory = cold_start_from_ledger(ledger, only_trustworthy=False)
    assert "tool" in memory.tool_stats


def test_cold_start_no_tools_uses_default_1():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"tool": (0.4, 5)}))
    memory = cold_start_from_ledger(ledger, tools=None, alpha=0.8)
    sr = memory.tool_stats["tool"]["success_rate"]
    # alpha=0.8: 0.8*0.4 + 0.2*1.0 = 0.52
    assert abs(sr - 0.52) < 0.05


def test_cold_start_returns_brain_memory():
    ledger = GlobalReputationLedger(min_orgs_required=1)
    ledger.ingest_snapshot(OrgReputationSnapshot("org_a", {"t": (0.9, 3)}))
    result = cold_start_from_ledger(ledger)
    assert isinstance(result, BrainMemory)
