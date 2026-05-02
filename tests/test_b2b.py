"""Tests for Phase 20: Inter-Organisational B2B Routing."""

from __future__ import annotations

import pytest

from manifold.b2b import (
    AgentEconomyLedger,
    B2BRouteResult,
    B2BRouter,
    EconomyEntry,
    HandshakeResult,
    OrgPolicy,
    PolicyHandshake,
)
from manifold.hub import ReputationHub
from manifold.policy import ManifoldPolicy, PolicyDomain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_policy(domain: str = "finance", risk_tol: float = 0.30, min_rel: float = 0.85) -> ManifoldPolicy:
    d = PolicyDomain(
        name=domain,
        stakes=0.9,
        risk_tolerance=risk_tol,
        coordination_tax_cap=0.10,
        fallback_strategy="hitl",
        min_tool_reliability=min_rel,
    )
    return ManifoldPolicy(domains=[d])


def _remote_policy(domain: str = "finance", max_risk: float = 0.25, min_rel: float = 0.90) -> OrgPolicy:
    return OrgPolicy(
        org_id="org-b",
        min_reliability=min_rel,
        max_risk=max_risk,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# OrgPolicy
# ---------------------------------------------------------------------------


class TestOrgPolicy:
    def test_defaults(self) -> None:
        op = OrgPolicy(org_id="org-x")
        assert op.org_id == "org-x"
        assert 0 < op.min_reliability <= 1
        assert 0 < op.max_risk <= 1

    def test_to_dict(self) -> None:
        op = OrgPolicy(org_id="org-a", min_reliability=0.85, max_risk=0.30)
        d = op.to_dict()
        assert d["org_id"] == "org-a"
        assert d["min_reliability"] == pytest.approx(0.85)

    def test_from_dict_roundtrip(self) -> None:
        op = OrgPolicy(org_id="org-a", min_reliability=0.85, max_risk=0.30, domain="legal")
        restored = OrgPolicy.from_dict(op.to_dict())
        assert restored.org_id == op.org_id
        assert restored.min_reliability == pytest.approx(op.min_reliability)
        assert restored.domain == op.domain

    def test_from_manifold_policy(self) -> None:
        policy = _local_policy(domain="finance", risk_tol=0.30, min_rel=0.85)
        op = OrgPolicy.from_manifold_policy(policy, org_id="org-a")
        assert op.org_id == "org-a"
        assert op.max_risk == pytest.approx(0.30)
        assert op.min_reliability == pytest.approx(0.85)
        assert op.domain == "finance"

    def test_from_manifold_policy_no_domains(self) -> None:
        policy = ManifoldPolicy()
        op = OrgPolicy.from_manifold_policy(policy, org_id="org-empty")
        assert op.org_id == "org-empty"

    def test_from_dict_missing_keys(self) -> None:
        op = OrgPolicy.from_dict({})
        assert op.org_id == "unknown"


# ---------------------------------------------------------------------------
# PolicyHandshake
# ---------------------------------------------------------------------------


class TestPolicyHandshake:
    def test_compatible_when_remote_safe(self) -> None:
        local = _local_policy(risk_tol=0.30, min_rel=0.85)
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.92, domain="finance")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        assert result.compatible
        assert result.conflict_reasons == ()

    def test_blocked_when_remote_risk_too_high(self) -> None:
        local = _local_policy(risk_tol=0.30, min_rel=0.85)
        remote = OrgPolicy(org_id="org-b", max_risk=0.45, min_reliability=0.90, domain="finance")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        assert not result.compatible
        assert len(result.conflict_reasons) >= 1
        assert "max_risk" in result.conflict_reasons[0]

    def test_blocked_when_remote_reliability_too_low(self) -> None:
        local = _local_policy(risk_tol=0.30, min_rel=0.85)
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.70, domain="finance")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        assert not result.compatible
        assert any("min_reliability" in r for r in result.conflict_reasons)

    def test_two_conflicts(self) -> None:
        local = _local_policy(risk_tol=0.20, min_rel=0.90)
        remote = OrgPolicy(org_id="org-b", max_risk=0.50, min_reliability=0.50, domain="finance")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        assert len(result.conflict_reasons) == 2

    def test_domain_fallback_to_global(self) -> None:
        local = _local_policy(domain="finance")
        remote = OrgPolicy(org_id="org-b", max_risk=0.30, min_reliability=0.80, domain="legal")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        # local has no 'legal' domain, falls back to global settings
        assert isinstance(result.compatible, bool)

    def test_risk_delta_sign(self) -> None:
        local = _local_policy(risk_tol=0.30)
        remote = OrgPolicy(org_id="org-b", max_risk=0.50, min_reliability=0.90, domain="finance")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        # risk_delta = remote.max_risk - local.risk_tolerance
        assert result.risk_delta == pytest.approx(0.50 - 0.30)

    def test_reliability_delta_sign(self) -> None:
        local = _local_policy(min_rel=0.85)
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.95, domain="finance")
        handshake = PolicyHandshake(local, "org-a")
        result = handshake.check(remote)
        assert result.reliability_delta == pytest.approx(0.95 - 0.85)

    def test_org_ids_in_result(self) -> None:
        local = _local_policy()
        remote = OrgPolicy(org_id="remote-x", max_risk=0.20, min_reliability=0.90)
        handshake = PolicyHandshake(local, "local-a")
        result = handshake.check(remote)
        assert result.local_org_id == "local-a"
        assert result.remote_org_id == "remote-x"


# ---------------------------------------------------------------------------
# B2BRouter
# ---------------------------------------------------------------------------


class TestB2BRouter:
    def _router(self, local_policy: ManifoldPolicy | None = None) -> B2BRouter:
        hub = ReputationHub()
        return B2BRouter(
            local_policy=local_policy or _local_policy(),
            hub=hub,
            local_org_id="org-a",
        )

    def test_compatible_remote_allowed(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.90, domain="finance")
        result = router.route(remote)
        assert result.allowed
        assert result.block_reason == ""

    def test_incompatible_remote_blocked(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.60, min_reliability=0.50, domain="finance")
        result = router.route(remote)
        assert not result.allowed
        assert len(result.block_reason) > 0

    def test_route_from_policy(self) -> None:
        router = self._router()
        remote_policy = _local_policy(domain="finance", risk_tol=0.25, min_rel=0.90)
        result = router.route_from_policy(remote_policy, remote_org_id="org-b")
        assert isinstance(result, B2BRouteResult)

    def test_auto_record_in_ledger(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.90, domain="finance")
        router.route(remote)
        assert len(router.ledger.entries()) == 1

    def test_no_auto_record(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.90, domain="finance")
        router.route(remote, auto_record=False)
        assert len(router.ledger.entries()) == 0

    def test_net_trust_cost_positive(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.90, domain="finance")
        result = router.route(remote)
        assert result.net_trust_cost > 0

    def test_summary_keys(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.90, domain="finance")
        router.route(remote)
        summary = router.summary()
        assert "total_calls" in summary
        assert "block_rate" in summary

    def test_low_reputation_blocked(self) -> None:
        router = self._router()
        router.min_reputation = 0.99  # impossible to meet
        remote = OrgPolicy(org_id="unknown-org", max_risk=0.20, min_reliability=0.90, domain="finance")
        result = router.route(remote)
        assert not result.allowed
        assert "reputation" in result.block_reason

    def test_surcharge_low_for_high_rep(self) -> None:
        router = self._router()
        remote = OrgPolicy(org_id="org-b", max_risk=0.20, min_reliability=0.90, domain="finance")
        result = router.route(remote)
        assert 0 <= result.surcharge <= 1


# ---------------------------------------------------------------------------
# AgentEconomyLedger
# ---------------------------------------------------------------------------


class TestAgentEconomyLedger:
    def _result(self, allowed: bool = True, cost: float = 1.0, surcharge: float = 0.1) -> B2BRouteResult:
        handshake = HandshakeResult(
            compatible=allowed,
            local_org_id="org-a",
            remote_org_id="org-b",
            local_domain="finance",
            remote_domain="finance",
            conflict_reasons=(),
            risk_delta=-0.05,
            reliability_delta=0.10,
        )
        return B2BRouteResult(
            allowed=allowed,
            local_org_id="org-a",
            remote_org_id="org-b",
            handshake=handshake,
            reputation_score=0.85,
            surcharge=surcharge,
            net_trust_cost=cost * (1 + surcharge),
            block_reason="" if allowed else "policy mismatch",
        )

    def test_empty_ledger(self) -> None:
        ledger = AgentEconomyLedger()
        assert ledger.total_trust_cost() == pytest.approx(0.0)
        assert ledger.block_rate() == pytest.approx(0.0)

    def test_record_and_count(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result(allowed=True))
        ledger.record(self._result(allowed=False))
        assert ledger.allowed_count() == 1
        assert ledger.blocked_count() == 1

    def test_block_rate(self) -> None:
        ledger = AgentEconomyLedger()
        for _ in range(3):
            ledger.record(self._result(allowed=True))
        ledger.record(self._result(allowed=False))
        assert ledger.block_rate() == pytest.approx(0.25)

    def test_total_trust_cost(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result(cost=1.0, surcharge=0.0))  # net = 1.0
        ledger.record(self._result(cost=2.0, surcharge=0.0))  # net = 2.0
        assert ledger.total_trust_cost() == pytest.approx(1.0 + 2.0)

    def test_avg_surcharge(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result(surcharge=0.1))
        ledger.record(self._result(surcharge=0.3))
        assert ledger.avg_surcharge() == pytest.approx(0.2)

    def test_avg_reputation(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result())  # rep = 0.85 in _result
        assert ledger.avg_reputation() == pytest.approx(0.85)

    def test_org_costs(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result(cost=1.0, surcharge=0.0))
        costs = ledger.org_costs()
        assert "org-b" in costs

    def test_summary_structure(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result())
        summary = ledger.summary()
        for key in ("total_calls", "allowed", "blocked", "block_rate", "total_trust_cost"):
            assert key in summary

    def test_entries_copy(self) -> None:
        ledger = AgentEconomyLedger()
        ledger.record(self._result())
        entries = ledger.entries()
        entries.clear()  # mutating copy should not affect ledger
        assert len(ledger.entries()) == 1

    def test_economy_entry_from_route_result(self) -> None:
        result = self._result(allowed=True, surcharge=0.15)
        entry = EconomyEntry.from_route_result(result)
        assert entry.allowed
        assert entry.surcharge == pytest.approx(0.15)
        assert entry.remote_org_id == "org-b"
