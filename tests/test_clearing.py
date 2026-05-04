"""Tests for Phase 32: Trust Clearinghouse (manifold/clearing.py)."""

from __future__ import annotations

import pytest

from manifold.b2b import AgentEconomyLedger, EconomyEntry
from manifold.clearing import (
    BankruptcyFreeze,
    ClearingEngine,
    SettlementEvent,
    SystemConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    local_org: str,
    remote_org: str,
    cost: float,
    allowed: bool = True,
) -> EconomyEntry:
    return EconomyEntry(
        local_org_id=local_org,
        remote_org_id=remote_org,
        allowed=allowed,
        reputation_score=0.8,
        surcharge=0.1,
        net_trust_cost=cost,
        block_reason="",
    )


def _ledger_with_entries(*entries: tuple[str, str, float]) -> AgentEconomyLedger:
    ledger = AgentEconomyLedger()
    for local, remote, cost in entries:
        ledger._entries.append(_make_entry(local, remote, cost))  # noqa: SLF001
    return ledger


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------


class TestSystemConfig:
    def test_defaults(self) -> None:
        cfg = SystemConfig()
        assert cfg.max_trust_debt == 50.0
        assert cfg.mint_reward == 5.0

    def test_custom_values(self) -> None:
        cfg = SystemConfig(max_trust_debt=100.0, mint_reward=10.0)
        assert cfg.max_trust_debt == 100.0
        assert cfg.mint_reward == 10.0

    def test_frozen(self) -> None:
        cfg = SystemConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_trust_debt = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SettlementEvent
# ---------------------------------------------------------------------------


class TestSettlementEvent:
    def test_creation(self) -> None:
        ev = SettlementEvent(
            timestamp=1_000_000.0,
            from_org="org-a",
            to_org="org-b",
            gross_forward=10.0,
            gross_reverse=4.0,
            net_amount=6.0,
            settled=True,
        )
        assert ev.from_org == "org-a"
        assert ev.to_org == "org-b"
        assert ev.net_amount == 6.0
        assert ev.settled is True

    def test_to_dict(self) -> None:
        ev = SettlementEvent(1_000_000.0, "a", "b", 5.0, 3.0, 2.0, True)
        d = ev.to_dict()
        assert d["from_org"] == "a"
        assert d["net_amount"] == 2.0
        assert d["settled"] is True

    def test_frozen(self) -> None:
        ev = SettlementEvent(0.0, "a", "b", 1.0, 0.0, 1.0, True)
        with pytest.raises((AttributeError, TypeError)):
            ev.net_amount = 99.0  # type: ignore[misc]

    def test_unsettled_flag(self) -> None:
        ev = SettlementEvent(0.0, "a", "b", 100.0, 0.0, 100.0, False)
        assert ev.settled is False


# ---------------------------------------------------------------------------
# BankruptcyFreeze
# ---------------------------------------------------------------------------


class TestBankruptcyFreeze:
    def test_creation(self) -> None:
        freeze = BankruptcyFreeze(org_id="bad-org", debt=99.0, timestamp=12345.0)
        assert freeze.org_id == "bad-org"
        assert freeze.debt == 99.0

    def test_to_dict(self) -> None:
        freeze = BankruptcyFreeze("x", 55.0, 9999.0)
        d = freeze.to_dict()
        assert d["org_id"] == "x"
        assert d["debt"] == 55.0

    def test_frozen(self) -> None:
        freeze = BankruptcyFreeze("x", 1.0, 0.0)
        with pytest.raises((AttributeError, TypeError)):
            freeze.debt = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ClearingEngine — settle()
# ---------------------------------------------------------------------------


class TestClearingEngineSettle:
    def test_empty_ledger_no_settlements(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        events = engine.settle()
        assert events == []

    def test_single_pair_forward_only(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 5.0))
        engine = ClearingEngine(ledger=ledger, clock=lambda: 1_000.0)
        events = engine.settle()
        assert len(events) == 1
        ev = events[0]
        assert ev.from_org == "org-a"
        assert ev.to_org == "org-b"
        assert ev.gross_forward == 5.0
        assert ev.gross_reverse == 0.0
        assert ev.net_amount == 5.0
        assert ev.settled is True

    def test_single_pair_bilateral_netting(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 8.0),
            ("org-b", "org-a", 3.0),
        )
        engine = ClearingEngine(ledger=ledger)
        events = engine.settle()
        assert len(events) == 1
        assert events[0].net_amount == pytest.approx(5.0)

    def test_net_amount_can_be_negative(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 2.0),
            ("org-b", "org-a", 6.0),
        )
        engine = ClearingEngine(ledger=ledger)
        events = engine.settle()
        # org-a < org-b lexicographically → forward is org-a→org-b
        assert events[0].net_amount == pytest.approx(-4.0)

    def test_multiple_pairs(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 5.0),
            ("org-a", "org-c", 3.0),
            ("org-b", "org-c", 2.0),
        )
        engine = ClearingEngine(ledger=ledger)
        events = engine.settle()
        # 3 unique pairs: (a,b), (a,c), (b,c)
        assert len(events) == 3

    def test_settlements_accumulated(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 5.0))
        engine = ClearingEngine(ledger=ledger)
        engine.settle()
        engine.settle()
        assert len(engine.settlements()) == 2

    def test_settled_flag_false_when_exceeds_cap(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 200.0))
        cfg = SystemConfig(max_trust_debt=50.0)
        engine = ClearingEngine(ledger=ledger, config=cfg)
        events = engine.settle()
        assert events[0].settled is False

    def test_settled_flag_true_within_cap(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 10.0))
        cfg = SystemConfig(max_trust_debt=50.0)
        engine = ClearingEngine(ledger=ledger, config=cfg)
        events = engine.settle()
        assert events[0].settled is True

    def test_same_org_pairs_not_double_counted(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 2.0),
            ("org-a", "org-b", 3.0),
        )
        engine = ClearingEngine(ledger=ledger)
        events = engine.settle()
        assert len(events) == 1
        assert events[0].gross_forward == pytest.approx(5.0)

    def test_returns_new_events_each_call(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 5.0))
        engine = ClearingEngine(ledger=ledger)
        e1 = engine.settle()
        e2 = engine.settle()
        assert e1 != e2  # separate list objects
        assert len(e1) == 1
        assert len(e2) == 1


# ---------------------------------------------------------------------------
# ClearingEngine — check_bankruptcy()
# ---------------------------------------------------------------------------


class TestClearingEngineBankruptcy:
    def test_no_freezes_below_cap(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 10.0))
        engine = ClearingEngine(ledger=ledger, config=SystemConfig(max_trust_debt=50.0))
        assert engine.check_bankruptcy() == []

    def test_freeze_emitted_above_cap(self) -> None:
        ledger = _ledger_with_entries(
            ("big-spender", "org-b", 30.0),
            ("big-spender", "org-c", 30.0),
        )
        engine = ClearingEngine(ledger=ledger, config=SystemConfig(max_trust_debt=50.0))
        freezes = engine.check_bankruptcy()
        assert len(freezes) == 1
        assert freezes[0].org_id == "big-spender"
        assert freezes[0].debt == pytest.approx(60.0)

    def test_freeze_accumulated_in_engine(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 100.0))
        engine = ClearingEngine(ledger=ledger, config=SystemConfig(max_trust_debt=50.0))
        engine.check_bankruptcy()
        engine.check_bankruptcy()
        assert len(engine.freezes()) == 2

    def test_multiple_bankrupt_orgs(self) -> None:
        ledger = _ledger_with_entries(
            ("org-x", "org-z", 100.0),
            ("org-y", "org-z", 80.0),
        )
        engine = ClearingEngine(ledger=ledger, config=SystemConfig(max_trust_debt=50.0))
        freezes = engine.check_bankruptcy()
        bankrupt_orgs = {f.org_id for f in freezes}
        assert "org-x" in bankrupt_orgs
        assert "org-y" in bankrupt_orgs

    def test_freeze_timestamp_from_clock(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 100.0))
        engine = ClearingEngine(
            ledger=ledger,
            config=SystemConfig(max_trust_debt=5.0),
            clock=lambda: 999.0,
        )
        freezes = engine.check_bankruptcy()
        assert freezes[0].timestamp == 999.0


# ---------------------------------------------------------------------------
# ClearingEngine — net_debt() / all_net_debts()
# ---------------------------------------------------------------------------


class TestClearingEngineNetDebt:
    def test_net_debt_payer(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 10.0),
            ("org-b", "org-a", 3.0),
        )
        engine = ClearingEngine(ledger=ledger)
        assert engine.net_debt("org-a") == pytest.approx(7.0)

    def test_net_debt_receiver(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 10.0),
            ("org-b", "org-a", 3.0),
        )
        engine = ClearingEngine(ledger=ledger)
        # org-b outbound=3, inbound=10 → net = -7
        assert engine.net_debt("org-b") == pytest.approx(-7.0)

    def test_net_debt_unknown_org(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        assert engine.net_debt("unknown") == 0.0

    def test_all_net_debts(self) -> None:
        ledger = _ledger_with_entries(
            ("org-a", "org-b", 5.0),
        )
        engine = ClearingEngine(ledger=ledger)
        debts = engine.all_net_debts()
        assert "org-a" in debts
        assert "org-b" in debts


# ---------------------------------------------------------------------------
# ClearingEngine — mint_for_canary_success()
# ---------------------------------------------------------------------------


class TestClearingEngineMint:
    def test_initial_balance_zero(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        assert engine.trust_balance("org-a") == 0.0

    def test_mint_adds_reward(self) -> None:
        cfg = SystemConfig(mint_reward=5.0)
        engine = ClearingEngine(ledger=AgentEconomyLedger(), config=cfg)
        bal = engine.mint_for_canary_success("org-a")
        assert bal == 5.0
        assert engine.trust_balance("org-a") == 5.0

    def test_mint_accumulates(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        engine.mint_for_canary_success("org-a")
        bal = engine.mint_for_canary_success("org-a")
        assert bal == pytest.approx(10.0)

    def test_different_orgs_independent(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        engine.mint_for_canary_success("org-a")
        engine.mint_for_canary_success("org-b")
        assert engine.trust_balance("org-a") == engine.trust_balance("org-b")


# ---------------------------------------------------------------------------
# ClearingEngine — summary()
# ---------------------------------------------------------------------------


class TestClearingEngineSummary:
    def test_empty_summary(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        s = engine.summary()
        assert s["total_settlements"] == 0
        assert s["total_freezes"] == 0
        assert s["net_debts"] == {}

    def test_summary_after_settle(self) -> None:
        ledger = _ledger_with_entries(("org-a", "org-b", 5.0))
        engine = ClearingEngine(ledger=ledger)
        engine.settle()
        s = engine.summary()
        assert s["total_settlements"] == 1

    def test_summary_trust_balances(self) -> None:
        engine = ClearingEngine(ledger=AgentEconomyLedger())
        engine.mint_for_canary_success("org-x")
        s = engine.summary()
        assert "org-x" in s["trust_balances"]
