"""Tests for Phase 22: Fleet Dashboard helpers."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from manifold.b2b import AgentEconomyLedger, B2BRouter, OrgPolicy
from manifold.gitops import CIRiskDelta, CIRiskReport
from manifold.hub import ReputationHub
from manifold.policy import ManifoldPolicy, PolicyDomain
from manifold.fleet import (
    B2BEconomySnapshot,
    CIBuildHistory,
    CIBuildRecord,
    FleetDashboardData,
    FleetPanelRenderer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ci_report(passed: bool = True, flagged: int = 0, max_delta: float = 0.05) -> CIRiskReport:
    deltas = []
    for i in range(flagged):
        deltas.append(
            CIRiskDelta(
                tool_name=f"tool-{i}",
                old_risk=0.10,
                new_risk=0.10 + max_delta,
                delta=max_delta,
                old_reliability=0.90,
                new_reliability=0.85,
                reliability_delta=-0.05,
                vetoed=True,
            )
        )
    vetoed = tuple(d.tool_name for d in deltas)
    return CIRiskReport(
        passed=passed,
        deltas=tuple(deltas),
        vetoed_tools=vetoed,
        risk_veto_threshold=0.15,
        max_reliability_drop=0.05,
        summary="test summary",
    )


def _build_record(
    build_id: str = "build-001",
    passed: bool = True,
    flagged: int = 0,
    max_delta: float = 0.05,
    branch: str = "main",
) -> CIBuildRecord:
    return CIBuildRecord(
        build_id=build_id,
        passed=passed,
        risk_report=_ci_report(passed=passed, flagged=flagged, max_delta=max_delta),
        branch=branch,
    )


def _simple_ledger(n_allowed: int = 3, n_blocked: int = 1) -> AgentEconomyLedger:
    from manifold.b2b import B2BRouteResult, HandshakeResult

    def _hs(compatible: bool) -> HandshakeResult:
        return HandshakeResult(
            compatible=compatible,
            local_org_id="local",
            remote_org_id="remote",
            local_domain="general",
            remote_domain="general",
            conflict_reasons=() if compatible else ("risk too high",),
            risk_delta=0.0,
            reliability_delta=0.0,
        )

    ledger = AgentEconomyLedger()
    for i in range(n_allowed):
        result = B2BRouteResult(
            allowed=True,
            local_org_id="org-a",
            remote_org_id=f"org-{i}",
            handshake=_hs(True),
            reputation_score=0.90,
            surcharge=0.01,
            net_trust_cost=1.01,
            block_reason="",
        )
        ledger.record(result)
    for i in range(n_blocked):
        result = B2BRouteResult(
            allowed=False,
            local_org_id="org-a",
            remote_org_id=f"bad-org-{i}",
            handshake=_hs(False),
            reputation_score=0.30,
            surcharge=0.49,
            net_trust_cost=1.49,
            block_reason="risk too high",
        )
        ledger.record(result)
    return ledger


# ---------------------------------------------------------------------------
# CIBuildRecord
# ---------------------------------------------------------------------------


class TestCIBuildRecord:
    def test_creation(self) -> None:
        r = _build_record()
        assert r.build_id == "build-001"
        assert r.passed is True
        assert r.branch == "main"

    def test_default_timestamp_is_utc(self) -> None:
        r = _build_record()
        assert r.timestamp.tzinfo is not None

    def test_to_dict_keys(self) -> None:
        r = _build_record()
        d = r.to_dict()
        assert "build_id" in d
        assert "passed" in d
        assert "max_delta" in d
        assert "flagged_tools" in d
        assert "timestamp" in d

    def test_to_dict_values(self) -> None:
        r = _build_record(build_id="b42", passed=False, flagged=2, max_delta=0.20)
        d = r.to_dict()
        assert d["build_id"] == "b42"
        assert d["passed"] is False
        assert d["flagged_tools"] == 2


# ---------------------------------------------------------------------------
# CIBuildHistory
# ---------------------------------------------------------------------------


class TestCIBuildHistory:
    def test_empty_history(self) -> None:
        h = CIBuildHistory()
        assert h.total_builds() == 0
        assert h.pass_rate() == 1.0

    def test_add_and_records(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1"))
        h.add(_build_record("b2"))
        assert h.total_builds() == 2
        assert len(h.records()) == 2

    def test_pass_rate_all_pass(self) -> None:
        h = CIBuildHistory()
        for i in range(5):
            h.add(_build_record(f"b{i}", passed=True))
        assert h.pass_rate() == pytest.approx(1.0)

    def test_pass_rate_mixed(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1", passed=True))
        h.add(_build_record("b2", passed=False))
        assert h.pass_rate() == pytest.approx(0.5)

    def test_failed_builds(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1", passed=True))
        h.add(_build_record("b2", passed=False))
        h.add(_build_record("b3", passed=False))
        failed = h.failed_builds()
        assert len(failed) == 2
        assert all(not r.passed for r in failed)

    def test_latest_newest_first(self) -> None:
        h = CIBuildHistory()
        for i in range(5):
            h.add(_build_record(f"b{i}"))
        latest = h.latest(3)
        assert len(latest) == 3
        assert latest[0].build_id == "b4"
        assert latest[1].build_id == "b3"

    def test_most_risky_tools(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1", flagged=2, max_delta=0.20))
        h.add(_build_record("b2", flagged=1, max_delta=0.10))
        risky = h.most_risky_tools(top_n=5)
        assert len(risky) > 0
        deltas = [d for _, d in risky]
        assert deltas == sorted(deltas, reverse=True)

    def test_avg_flagged_per_build(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1", flagged=2))
        h.add(_build_record("b2", flagged=4))
        assert h.avg_flagged_per_build() == pytest.approx(3.0)

    def test_summary_keys(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1", passed=True))
        h.add(_build_record("b2", passed=False))
        s = h.summary()
        assert s["total_builds"] == 2
        assert s["passed"] == 1
        assert s["failed"] == 1
        assert "pass_rate" in s
        assert "avg_flagged_per_build" in s

    def test_most_risky_empty(self) -> None:
        h = CIBuildHistory()
        assert h.most_risky_tools() == []


# ---------------------------------------------------------------------------
# B2BEconomySnapshot
# ---------------------------------------------------------------------------


class TestB2BEconomySnapshot:
    def test_empty_snapshot(self) -> None:
        snap = B2BEconomySnapshot()
        assert snap.total_trust_cost() == pytest.approx(0.0)
        assert snap.block_rate() == pytest.approx(0.0)
        assert snap.avg_reputation() == pytest.approx(0.0)

    def test_from_ledgers(self) -> None:
        ledger = _simple_ledger(n_allowed=3, n_blocked=1)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        assert snap.total_trust_cost() > 0

    def test_from_multiple_ledgers(self) -> None:
        l1 = _simple_ledger(n_allowed=2, n_blocked=0)
        l2 = _simple_ledger(n_allowed=1, n_blocked=1)
        snap = B2BEconomySnapshot.from_ledgers([l1, l2])
        assert len(snap.entries) == 4

    def test_block_rate(self) -> None:
        ledger = _simple_ledger(n_allowed=3, n_blocked=1)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        assert snap.block_rate() == pytest.approx(0.25)

    def test_org_costs(self) -> None:
        ledger = _simple_ledger(n_allowed=2, n_blocked=0)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        costs = snap.org_costs()
        assert len(costs) >= 1
        assert all(v > 0 for v in costs.values())

    def test_top_partners(self) -> None:
        ledger = _simple_ledger(n_allowed=3, n_blocked=0)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        partners = snap.top_partners(top_n=5)
        assert len(partners) >= 1
        costs = [c for _, c in partners]
        assert costs == sorted(costs, reverse=True)

    def test_org_block_rates(self) -> None:
        ledger = _simple_ledger(n_allowed=3, n_blocked=1)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        rates = snap.org_block_rates()
        assert all(0.0 <= v <= 1.0 for v in rates.values())

    def test_summary_keys(self) -> None:
        ledger = _simple_ledger(n_allowed=2, n_blocked=1)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        s = snap.summary()
        assert "total_calls" in s
        assert "block_rate" in s
        assert "total_trust_cost" in s
        assert "avg_reputation" in s
        assert "unique_remote_orgs" in s

    def test_avg_reputation(self) -> None:
        ledger = _simple_ledger(n_allowed=4, n_blocked=0)
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        assert 0.0 < snap.avg_reputation() <= 1.0

    def test_from_routers(self) -> None:
        policy = ManifoldPolicy(domains=[PolicyDomain(name="general")])
        hub = ReputationHub()
        router = B2BRouter(local_policy=policy, hub=hub, local_org_id="org-a")
        remote = OrgPolicy(org_id="org-b", min_reliability=0.80, max_risk=0.30, domain="general")
        router.route(remote)
        snap = B2BEconomySnapshot.from_routers([router])
        assert len(snap.entries) == 1


# ---------------------------------------------------------------------------
# FleetDashboardData
# ---------------------------------------------------------------------------


class TestFleetDashboardData:
    def test_defaults(self) -> None:
        data = FleetDashboardData()
        assert data.node_id == "manifold-node"
        assert data.version == "1.1.0"

    def test_to_summary_dict(self) -> None:
        data = FleetDashboardData(node_id="prod-1")
        d = data.to_summary_dict()
        assert d["node_id"] == "prod-1"
        assert "ci" in d
        assert "economy" in d
        assert d["version"] == "1.1.0"

    def test_summary_dict_ci_keys(self) -> None:
        h = CIBuildHistory()
        h.add(_build_record("b1"))
        data = FleetDashboardData(ci_history=h)
        d = data.to_summary_dict()
        assert "total_builds" in d["ci"]

    def test_summary_dict_economy_keys(self) -> None:
        ledger = _simple_ledger()
        snap = B2BEconomySnapshot.from_ledgers([ledger])
        data = FleetDashboardData(economy=snap)
        d = data.to_summary_dict()
        assert "total_calls" in d["economy"]


# ---------------------------------------------------------------------------
# FleetPanelRenderer
# ---------------------------------------------------------------------------


class TestFleetPanelRenderer:
    def _data(self) -> FleetDashboardData:
        h = CIBuildHistory()
        h.add(_build_record("b1", passed=True))
        h.add(_build_record("b2", passed=False, flagged=1, max_delta=0.18))
        ledger = _simple_ledger(n_allowed=3, n_blocked=1)
        snap = B2BEconomySnapshot.from_ledgers([ledger], org_labels={"org-0": "Acme Corp"})
        return FleetDashboardData(ci_history=h, economy=snap, node_id="fleet-test")

    def test_ci_summary_text_contains_node_id(self) -> None:
        r = FleetPanelRenderer(self._data())
        text = r.ci_summary_text()
        assert "fleet-test" in text

    def test_ci_summary_text_contains_pass_rate(self) -> None:
        r = FleetPanelRenderer(self._data())
        text = r.ci_summary_text()
        assert "Pass rate" in text or "pass_rate" in text.lower()

    def test_economy_summary_text_contains_block_rate(self) -> None:
        r = FleetPanelRenderer(self._data())
        text = r.economy_summary_text()
        assert "Block rate" in text or "block_rate" in text.lower()

    def test_economy_summary_text_contains_org_label(self) -> None:
        r = FleetPanelRenderer(self._data())
        text = r.economy_summary_text()
        # org-0 is mapped to "Acme Corp"
        assert "Acme Corp" in text

    def test_full_report_combines_both(self) -> None:
        r = FleetPanelRenderer(self._data())
        full = r.full_report_text()
        assert "CI/CD" in full
        assert "B2B" in full

    def test_empty_data_no_crash(self) -> None:
        r = FleetPanelRenderer(FleetDashboardData())
        # Should not raise
        text = r.full_report_text()
        assert isinstance(text, str)
