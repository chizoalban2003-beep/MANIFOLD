"""Tests for Phase 15: Global Reputation Hub (CommunityBaseline + ReputationHub)."""

from __future__ import annotations

import pytest

from manifold import FederatedGossipPacket, GlobalReputationLedger, OrgReputationSnapshot
from manifold.hub import CommunityBaseline, ReputationHub


# ---------------------------------------------------------------------------
# CommunityBaseline tests
# ---------------------------------------------------------------------------

class TestCommunityBaseline:
    def test_default_loads(self):
        baseline = CommunityBaseline.default()
        assert isinstance(baseline, CommunityBaseline)
        assert len(baseline.tool_baselines) > 0

    def test_default_contains_common_tools(self):
        baseline = CommunityBaseline.default()
        assert "gpt-4o" in baseline.tool_baselines
        assert "gpt-4o-mini" in baseline.tool_baselines
        assert "claude-3-5-sonnet" in baseline.tool_baselines
        assert "calculator" in baseline.tool_baselines

    def test_reliability_known_tool(self):
        baseline = CommunityBaseline.default()
        r = baseline.reliability("gpt-4o")
        assert r is not None
        assert 0.0 <= r <= 1.0

    def test_reliability_unknown_tool_returns_none(self):
        baseline = CommunityBaseline.default()
        assert baseline.reliability("completely_unknown_tool_xyz") is None

    def test_observation_weight_known_tool(self):
        baseline = CommunityBaseline.default()
        w = baseline.observation_weight("gpt-4o")
        assert w > 0

    def test_observation_weight_unknown_returns_zero(self):
        baseline = CommunityBaseline.default()
        assert baseline.observation_weight("unknown_xyz") == 0

    def test_is_flagged_known_risk(self):
        baseline = CommunityBaseline.default()
        # gpt-4o-mini has a known risk flag
        assert baseline.is_flagged("gpt-4o-mini") is True

    def test_is_flagged_unflagged_tool(self):
        baseline = CommunityBaseline.default()
        # calculator has no risk flag
        assert baseline.is_flagged("calculator") is False

    def test_risk_flag_returns_string(self):
        baseline = CommunityBaseline.default()
        flag = baseline.risk_flag("gpt-4o-mini")
        assert isinstance(flag, str)
        assert len(flag) > 0

    def test_risk_flag_unknown_returns_none(self):
        baseline = CommunityBaseline.default()
        assert baseline.risk_flag("unknown_xyz") is None

    def test_tool_names_sorted(self):
        baseline = CommunityBaseline.default()
        names = baseline.tool_names()
        assert names == sorted(names)

    def test_to_org_snapshot_produces_snapshot(self):
        baseline = CommunityBaseline.default()
        snap = baseline.to_org_snapshot()
        assert isinstance(snap, OrgReputationSnapshot)
        assert len(snap.rates) == len(baseline.tool_baselines)

    def test_to_org_snapshot_custom_org_id(self):
        baseline = CommunityBaseline.default()
        snap = baseline.to_org_snapshot(org_id="test_org")
        assert snap.org_id == "test_org"

    def test_reliability_clamped_to_01(self):
        """CommunityBaseline reliabilities must all be in [0, 1]."""
        baseline = CommunityBaseline.default()
        for name in baseline.tool_names():
            r = baseline.reliability(name)
            assert r is not None
            assert 0.0 <= r <= 1.0, f"{name} reliability out of range: {r}"

    def test_custom_baseline(self):
        baseline = CommunityBaseline(
            tool_baselines={"my_tool": (0.90, 100)},
            risk_flags={"my_tool": "Known latency spikes"},
            version="2.0.0",
        )
        assert baseline.reliability("my_tool") == pytest.approx(0.90)
        assert baseline.is_flagged("my_tool") is True
        assert baseline.version == "2.0.0"


# ---------------------------------------------------------------------------
# ReputationHub tests
# ---------------------------------------------------------------------------

class TestReputationHub:
    def test_instantiation_default(self):
        hub = ReputationHub()
        assert hub is not None

    def test_instantiation_custom_baseline(self):
        baseline = CommunityBaseline.default()
        hub = ReputationHub(baseline=baseline)
        assert hub.baseline is baseline

    def test_community_summary_keys(self):
        hub = ReputationHub()
        s = hub.community_summary()
        assert "baseline_tools" in s
        assert "flagged_tools" in s
        assert "total_contributions" in s
        assert "baseline_version" in s

    def test_community_summary_initial(self):
        hub = ReputationHub()
        s = hub.community_summary()
        assert s["total_contributions"] == 0
        assert s["baseline_tools"] > 0

    def test_contribute_increases_count(self):
        hub = ReputationHub()
        hub.contribute(FederatedGossipPacket(
            tool_name="gpt-4o", signal="failing", confidence=0.9, org_id="org_a"
        ))
        assert hub.contribution_count() == 1
        assert hub.contribution_count("gpt-4o") == 1

    def test_contribute_anonymize_strips_org_id(self):
        hub = ReputationHub()
        hub.contribute(
            FederatedGossipPacket(tool_name="tool_x", signal="failing", confidence=0.8, org_id="secret_org"),
            anonymize=True,
        )
        # Internal packets should have org_id replaced
        assert hub._contributions[0].org_id == "contributor"

    def test_contribute_no_anonymize_preserves_org_id(self):
        hub = ReputationHub()
        hub.contribute(
            FederatedGossipPacket(tool_name="tool_x", signal="healthy", confidence=0.8, org_id="public_org"),
            anonymize=False,
        )
        assert hub._contributions[0].org_id == "public_org"

    def test_warm_start_ledger_seeds_data(self):
        hub = ReputationHub()
        ledger = GlobalReputationLedger(min_orgs_required=1)
        hub.warm_start_ledger(ledger)
        # gpt-4o should now have a rate in the ledger
        rate = ledger.global_rate("gpt-4o")
        assert rate is not None
        assert 0.0 <= rate <= 1.0

    def test_warm_start_ledger_includes_contributions(self):
        hub = ReputationHub()
        hub.contribute(FederatedGossipPacket(
            tool_name="new_tool", signal="healthy", confidence=0.9, org_id="org_a"
        ))
        ledger = GlobalReputationLedger(min_orgs_required=1)
        hub.warm_start_ledger(ledger, include_contributions=True)
        rate = ledger.global_rate("new_tool")
        assert rate is not None

    def test_warm_start_ledger_excludes_contributions_when_flag_false(self):
        hub = ReputationHub()
        hub.contribute(FederatedGossipPacket(
            tool_name="exclusive_tool", signal="failing", confidence=0.9, org_id="org_a"
        ))
        ledger = GlobalReputationLedger(min_orgs_required=1)
        hub.warm_start_ledger(ledger, include_contributions=False)
        # exclusive_tool only came from contribution, not baseline
        rate = ledger.global_rate("exclusive_tool")
        assert rate is None  # not in baseline

    def test_live_reliability_known_tool(self):
        hub = ReputationHub()
        r = hub.live_reliability("gpt-4o")
        assert r is not None
        assert 0.0 <= r <= 1.0

    def test_live_reliability_unknown_tool_returns_none(self):
        hub = ReputationHub()
        r = hub.live_reliability("totally_unknown_tool_999")
        assert r is None

    def test_live_reliability_degrades_after_failures(self):
        hub = ReputationHub()
        initial = hub.live_reliability("gpt-4o")
        for _ in range(20):
            hub.contribute(FederatedGossipPacket(
                tool_name="gpt-4o", signal="failing", confidence=0.95, org_id="org_x"
            ))
        updated = hub.live_reliability("gpt-4o")
        # After 20 failure signals, reliability should decrease
        assert updated is not None
        assert updated < initial

    def test_flagged_tools_list(self):
        hub = ReputationHub()
        flagged = hub.flagged_tools()
        assert isinstance(flagged, list)
        assert len(flagged) > 0
        assert "gpt-4o-mini" in flagged

    def test_to_org_snapshot(self):
        hub = ReputationHub()
        snap = hub.to_org_snapshot()
        assert isinstance(snap, OrgReputationSnapshot)
        assert snap.org_id == "community_hub"
        assert len(snap.rates) > 0

    def test_to_org_snapshot_custom_org_id(self):
        hub = ReputationHub()
        snap = hub.to_org_snapshot(org_id="my_hub")
        assert snap.org_id == "my_hub"

    def test_contribution_count_per_tool(self):
        hub = ReputationHub()
        for _ in range(3):
            hub.contribute(FederatedGossipPacket(
                tool_name="tool_a", signal="healthy", confidence=0.8, org_id="o"
            ))
        hub.contribute(FederatedGossipPacket(
            tool_name="tool_b", signal="failing", confidence=0.9, org_id="o"
        ))
        assert hub.contribution_count("tool_a") == 3
        assert hub.contribution_count("tool_b") == 1
        assert hub.contribution_count() == 4

    def test_multiple_contributions_healthy_signal(self):
        hub = ReputationHub()
        for _ in range(10):
            hub.contribute(FederatedGossipPacket(
                tool_name="calculator", signal="healthy", confidence=0.99, org_id="org"
            ))
        r = hub.live_reliability("calculator")
        assert r is not None
        assert r >= 0.9  # calculator was already high; healthy signals keep it high

    def test_hub_integrates_with_federated_ledger(self):
        """Hub can warm-start a real GlobalReputationLedger that then rates all tools."""
        hub = ReputationHub()
        ledger = GlobalReputationLedger(min_orgs_required=1)
        hub.warm_start_ledger(ledger)
        # All baseline tools should now be ratable
        for name in hub.baseline.tool_names():
            rate = ledger.global_rate(name)
            assert rate is not None, f"{name} missing from ledger after warm_start"
