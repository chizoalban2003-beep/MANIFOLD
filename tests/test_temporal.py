"""Tests for Phase 63: Temporal Forking Engine (manifold/temporal.py).

Adversarial focus: forked timelines must NOT leak state or Trust Tokens
into the main ledger or into sibling branches.
"""

from __future__ import annotations

import time
import uuid

import pytest

from manifold.b2b import AgentEconomyLedger, B2BRouteResult, HandshakeResult
from manifold.gridmapper import GridState, GridWorld
from manifold.temporal import (
    BranchResult,
    CollapseResult,
    ForkSpec,
    ParallelTimeline,
    StateForker,
    TimelineCollapse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(size: int = 5) -> GridState:
    """Return a minimal GridState for testing."""
    world = GridWorld(size=size)
    return GridState(
        world=world,
        description="test",
        domain="test",
        parameters={"key": "value"},
        cell_profile=(0.1, 0.2, 0.3, 0.4),
    )


def _make_ledger() -> AgentEconomyLedger:
    return AgentEconomyLedger()


def _make_handshake() -> HandshakeResult:
    return HandshakeResult(
        compatible=True,
        local_org_id="org_a",
        remote_org_id="org_b",
        local_domain="general",
        remote_domain="general",
        conflict_reasons=(),
        risk_delta=0.0,
        reliability_delta=0.0,
    )


def _make_route_result(*, allowed: bool = True) -> B2BRouteResult:
    return B2BRouteResult(
        local_org_id="org_a",
        remote_org_id="org_b",
        allowed=allowed,
        reputation_score=0.8,
        surcharge=0.05,
        net_trust_cost=0.1,
        block_reason="",
        handshake=_make_handshake(),
    )


# ---------------------------------------------------------------------------
# ForkSpec
# ---------------------------------------------------------------------------


class TestForkSpec:
    def test_frozen(self) -> None:
        spec = ForkSpec(branch_id="b1", label="optimistic")
        with pytest.raises((AttributeError, TypeError)):
            spec.branch_id = "new"  # type: ignore[misc]

    def test_fields(self) -> None:
        spec = ForkSpec(branch_id="abc", label="conservative")
        assert spec.branch_id == "abc"
        assert spec.label == "conservative"

    def test_inequality(self) -> None:
        s1 = ForkSpec(branch_id="a", label="x")
        s2 = ForkSpec(branch_id="b", label="x")
        assert s1 != s2


# ---------------------------------------------------------------------------
# BranchResult
# ---------------------------------------------------------------------------


class TestBranchResult:
    def test_yield_score_normal(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=10.0, cost=2.0, risk=3.0, success=True
        )
        assert abs(r.yield_score() - 10.0 / 5.0) < 1e-9

    def test_yield_score_zero_denom(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=5.0, cost=0.0, risk=0.0, success=True
        )
        # Should not raise; denom is floored
        assert r.yield_score() > 0

    def test_yield_score_failed_branch(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=0.0, cost=0.0, risk=1.0, success=False
        )
        # risk=1 → denom=1 → score=0
        assert r.yield_score() == 0.0

    def test_to_dict_keys(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=1.0, cost=0.5, risk=0.5, success=True
        )
        d = r.to_dict()
        for key in ("branch_id", "label", "asset", "cost", "risk", "success",
                    "error", "duration_s", "yield_score"):
            assert key in d

    def test_to_dict_yield_score_value(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=4.0, cost=1.0, risk=1.0, success=True
        )
        d = r.to_dict()
        assert abs(d["yield_score"] - 2.0) < 1e-9

    def test_mutable(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=1.0, cost=0.5, risk=0.5, success=True
        )
        r.duration_s = 1.23
        assert r.duration_s == 1.23

    def test_error_default_empty(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=1.0, cost=0.5, risk=0.5, success=True
        )
        assert r.error == ""

    def test_success_false_captures_error(self) -> None:
        r = BranchResult(
            branch_id="b", label="l", asset=0.0, cost=0.0, risk=1.0,
            success=False, error="connection refused"
        )
        assert r.error == "connection refused"


# ---------------------------------------------------------------------------
# CollapseResult
# ---------------------------------------------------------------------------


class TestCollapseResult:
    def _make_result(self, asset: float, cost: float, risk: float, success: bool = True) -> BranchResult:
        return BranchResult(
            branch_id=str(uuid.uuid4()),
            label="x",
            asset=asset,
            cost=cost,
            risk=risk,
            success=success,
        )

    def test_frozen(self) -> None:
        r = self._make_result(1.0, 0.5, 0.5)
        c = CollapseResult(
            winning_branch_id="x",
            winning_label="x",
            winning_yield=1.0,
            all_branches=(r,),
            collapsed_at=time.time(),
            fork_id="f1",
        )
        with pytest.raises((AttributeError, TypeError)):
            c.fork_id = "new"  # type: ignore[misc]

    def test_to_dict_keys(self) -> None:
        r = self._make_result(1.0, 0.5, 0.5)
        c = CollapseResult(
            winning_branch_id=r.branch_id,
            winning_label=r.label,
            winning_yield=r.yield_score(),
            all_branches=(r,),
            collapsed_at=time.time(),
            fork_id="f1",
        )
        d = c.to_dict()
        for key in ("winning_branch_id", "winning_label", "winning_yield",
                    "all_branches", "collapsed_at", "fork_id"):
            assert key in d

    def test_all_branches_serialised(self) -> None:
        r1 = self._make_result(2.0, 0.5, 0.5)
        r2 = self._make_result(1.0, 0.5, 0.5)
        c = CollapseResult(
            winning_branch_id=r1.branch_id,
            winning_label=r1.label,
            winning_yield=r1.yield_score(),
            all_branches=(r1, r2),
            collapsed_at=time.time(),
            fork_id="f1",
        )
        d = c.to_dict()
        assert len(d["all_branches"]) == 2


# ---------------------------------------------------------------------------
# StateForker — isolation guarantees
# ---------------------------------------------------------------------------


class TestStateForker:
    def test_fork_grid_state_is_independent(self) -> None:
        state = _make_state()
        forked = StateForker.fork_grid_state(state)
        # Mutate the fork
        forked.world.set_cell(0, 0, 0.9, 0.9, 0.1, 0.5)
        # Original must be unchanged
        assert state.world.cells[0][0].cost != 0.9

    def test_fork_grid_state_different_object(self) -> None:
        state = _make_state()
        forked = StateForker.fork_grid_state(state)
        assert forked is not state
        assert forked.world is not state.world

    def test_fork_economy_ledger_is_independent(self) -> None:
        ledger = _make_ledger()
        ledger.record(_make_route_result())
        forked = StateForker.fork_economy_ledger(ledger)
        # Add entry to forked — original must not grow
        forked.record(_make_route_result(allowed=False))
        assert len(ledger.entries()) == 1
        assert len(forked.entries()) == 2

    def test_fork_economy_ledger_different_object(self) -> None:
        ledger = _make_ledger()
        forked = StateForker.fork_economy_ledger(ledger)
        assert forked is not ledger

    def test_fork_both_returns_tuple(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        s2, l2 = StateForker.fork_both(state, ledger)
        assert s2 is not state
        assert l2 is not ledger

    def test_fork_both_state_isolated(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        s2, _l2 = StateForker.fork_both(state, ledger)
        s2.world.set_cell(2, 2, 0.99, 0.99, 0.01, 0.0)
        assert state.world.cells[2][2].cost != 0.99

    def test_fork_both_ledger_isolated(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        _s2, l2 = StateForker.fork_both(state, ledger)
        l2.record(_make_route_result())
        assert len(ledger.entries()) == 0

    def test_fork_preserves_parameters(self) -> None:
        state = _make_state()
        forked = StateForker.fork_grid_state(state)
        assert forked.parameters == state.parameters

    def test_fork_preserves_cell_profile(self) -> None:
        state = _make_state()
        forked = StateForker.fork_grid_state(state)
        assert forked.cell_profile == state.cell_profile

    def test_multiple_forks_dont_share_refs(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        forks = [StateForker.fork_both(state, ledger) for _ in range(3)]
        # Mutate fork 0
        forks[0][0].world.set_cell(0, 0, 0.8, 0.8, 0.2, 0.0)
        # Fork 1 and 2 and original must be unchanged
        assert forks[1][0].world.cells[0][0].cost != 0.8
        assert forks[2][0].world.cells[0][0].cost != 0.8
        assert state.world.cells[0][0].cost != 0.8


# ---------------------------------------------------------------------------
# ParallelTimeline
# ---------------------------------------------------------------------------


class TestParallelTimeline:
    def test_add_branch_returns_spec(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger)
        spec = tl.add_branch("fast")
        assert isinstance(spec, ForkSpec)
        assert spec.label == "fast"

    def test_branch_count(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger)
        assert tl.branch_count == 0
        tl.add_branch("a")
        tl.add_branch("b")
        assert tl.branch_count == 2

    def test_fork_id_is_uuid(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger)
        uuid.UUID(tl.fork_id)  # raises if not valid UUID

    def test_custom_fork_id(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger, fork_id="my-fork-id")
        assert tl.fork_id == "my-fork-id"

    def test_run_returns_one_result_per_branch(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger)
        tl.add_branch("a")
        tl.add_branch("b")
        tl.add_branch("c")

        def executor(branch_id: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            return BranchResult(
                branch_id=branch_id, label="x", asset=1.0, cost=0.5, risk=0.5, success=True
            )

        results = tl.run(executor)
        assert len(results) == 3

    def test_run_no_branches_returns_empty(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger)
        results = tl.run(lambda bid, s, l: BranchResult(bid, "x", 1.0, 0.5, 0.5, True))
        assert results == []

    def test_branches_receive_independent_states(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        received_states: list[GridState] = []

        def executor(branch_id: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            received_states.append(s)
            return BranchResult(branch_id, "x", 1.0, 0.5, 0.5, True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("a")
        tl.add_branch("b")
        tl.run(executor)

        assert len(received_states) == 2
        # Each branch gets a different object
        assert received_states[0] is not received_states[1]
        assert received_states[0] is not state

    def test_branch_mutation_does_not_affect_source(self) -> None:
        state = _make_state()
        ledger = _make_ledger()

        def executor(branch_id: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            # Adversarial: try to modify the received grid
            s.world.set_cell(0, 0, 0.99, 0.99, 0.01, 0.0)
            return BranchResult(branch_id, "x", 1.0, 0.5, 0.5, True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("evil")
        tl.run(executor)

        # Source state must be unchanged
        assert state.world.cells[0][0].cost != 0.99

    def test_branch_trust_token_leak_prevention(self) -> None:
        """Adversarial: branch records entries in its forked ledger;
        the source ledger must remain empty."""
        state = _make_state()
        ledger = _make_ledger()

        def executor(branch_id: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            # Record in the forked ledger
            l.record(_make_route_result())
            l.record(_make_route_result(allowed=False))
            return BranchResult(
                branch_id, "x",
                asset=float(l.total_trust_cost()),
                cost=0.1, risk=0.1, success=True,
            )

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("spender-1")
        tl.add_branch("spender-2")
        tl.run(executor)

        # Source ledger must have ZERO entries
        assert len(ledger.entries()) == 0
        assert ledger.total_trust_cost() == 0.0

    def test_sibling_branches_dont_share_ledger(self) -> None:
        """Sibling branches cannot see each other's ledger entries."""
        state = _make_state()
        ledger = _make_ledger()
        entry_counts: list[int] = []

        def executor(branch_id: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            # Record one entry, then report how many entries we see
            l.record(_make_route_result())
            entry_counts.append(len(l.entries()))
            return BranchResult(branch_id, "x", 1.0, 0.5, 0.5, True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("b1")
        tl.add_branch("b2")
        tl.add_branch("b3")
        tl.run(executor)

        # Each branch should have seen exactly 1 entry (its own)
        assert all(c == 1 for c in entry_counts)

    def test_executor_exception_produces_failure_result(self) -> None:
        state = _make_state()
        ledger = _make_ledger()

        def bad_executor(bid: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            raise RuntimeError("deliberate failure")

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("bomb")
        results = tl.run(bad_executor)
        assert len(results) == 1
        assert results[0].success is False
        assert "deliberate failure" in results[0].error

    def test_failed_branch_does_not_stop_others(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        call_order: list[str] = []

        def executor(bid: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            call_order.append(bid)
            if len(call_order) == 1:
                raise RuntimeError("first branch crashes")
            return BranchResult(bid, "x", 1.0, 0.5, 0.5, True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("crash")
        tl.add_branch("ok")
        results = tl.run(executor)

        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is True

    def test_duration_is_set(self) -> None:
        state = _make_state()
        ledger = _make_ledger()

        def executor(bid: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            time.sleep(0.001)
            return BranchResult(bid, "x", 1.0, 0.5, 0.5, True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("timed")
        results = tl.run(executor)
        assert results[0].duration_s > 0.0

    def test_results_accessor_returns_copy(self) -> None:
        state = _make_state()
        ledger = _make_ledger()
        tl = ParallelTimeline(state, ledger)
        tl.add_branch("a")
        tl.run(lambda bid, s, l: BranchResult(bid, "x", 1.0, 0.5, 0.5, True))
        r1 = tl.results()
        r2 = tl.results()
        r1.append(BranchResult("fake", "x", 0.0, 0.0, 0.0, False))
        assert len(tl.results()) == 1  # internal list not modified


# ---------------------------------------------------------------------------
# TimelineCollapse
# ---------------------------------------------------------------------------


class TestTimelineCollapse:
    def _make_br(self, asset: float, cost: float, risk: float, success: bool = True) -> BranchResult:
        return BranchResult(
            branch_id=str(uuid.uuid4()),
            label=f"branch-{asset}",
            asset=asset,
            cost=cost,
            risk=risk,
            success=success,
        )

    def test_collapse_picks_highest_yield(self) -> None:
        b1 = self._make_br(asset=10.0, cost=1.0, risk=1.0)  # yield=5.0
        b2 = self._make_br(asset=4.0, cost=1.0, risk=1.0)   # yield=2.0
        b3 = self._make_br(asset=6.0, cost=1.0, risk=1.0)   # yield=3.0
        result = TimelineCollapse.collapse([b1, b2, b3])
        assert result.winning_branch_id == b1.branch_id

    def test_collapse_prefers_successful_branch(self) -> None:
        # Failed branch has a higher raw score but success=False
        b_fail = self._make_br(asset=100.0, cost=0.01, risk=0.01, success=False)
        b_pass = self._make_br(asset=5.0, cost=1.0, risk=1.0, success=True)
        result = TimelineCollapse.collapse([b_fail, b_pass])
        assert result.winning_branch_id == b_pass.branch_id

    def test_collapse_all_failed_picks_best_failed(self) -> None:
        b1 = self._make_br(asset=10.0, cost=1.0, risk=1.0, success=False)
        b2 = self._make_br(asset=2.0, cost=1.0, risk=1.0, success=False)
        result = TimelineCollapse.collapse([b1, b2])
        assert result.winning_branch_id == b1.branch_id

    def test_collapse_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            TimelineCollapse.collapse([])

    def test_collapse_single_branch(self) -> None:
        b = self._make_br(asset=3.0, cost=1.0, risk=1.0)
        result = TimelineCollapse.collapse([b], fork_id="fid")
        assert result.winning_branch_id == b.branch_id
        assert result.fork_id == "fid"

    def test_collapse_preserves_all_branches(self) -> None:
        branches = [self._make_br(float(i), 0.5, 0.5) for i in range(5)]
        result = TimelineCollapse.collapse(branches)
        assert len(result.all_branches) == 5

    def test_collapse_winning_yield_matches(self) -> None:
        b = self._make_br(asset=6.0, cost=1.0, risk=2.0)  # yield=2.0
        result = TimelineCollapse.collapse([b])
        assert abs(result.winning_yield - 2.0) < 1e-9

    def test_collapse_collapsed_at_is_recent(self) -> None:
        b = self._make_br(1.0, 0.5, 0.5)
        before = time.time()
        result = TimelineCollapse.collapse([b])
        after = time.time()
        assert before <= result.collapsed_at <= after

    def test_collapse_to_dict_roundtrip(self) -> None:
        b1 = self._make_br(3.0, 1.0, 1.0)
        b2 = self._make_br(1.0, 1.0, 1.0)
        result = TimelineCollapse.collapse([b1, b2], fork_id="test-fork")
        d = result.to_dict()
        assert d["fork_id"] == "test-fork"
        assert d["winning_branch_id"] == b1.branch_id

    def test_collapse_fork_id_propagated(self) -> None:
        b = self._make_br(1.0, 0.5, 0.5)
        result = TimelineCollapse.collapse([b], fork_id="my-fork-123")
        assert result.fork_id == "my-fork-123"


# ---------------------------------------------------------------------------
# Integration: full fork-run-collapse pipeline
# ---------------------------------------------------------------------------


class TestTemporalPipelineIntegration:
    def test_full_pipeline_no_state_leak(self) -> None:
        state = _make_state(size=7)
        ledger = _make_ledger()

        def executor(bid: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            s.world.set_cell(0, 0, 0.9, 0.9, 0.1, 0.0)
            l.record(_make_route_result())
            cost = 0.1 + float(s.world.cells[0][0].cost) * 0.1
            return BranchResult(bid, "branch", asset=5.0, cost=cost, risk=0.2, success=True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("optimistic")
        tl.add_branch("conservative")
        results = tl.run(executor)
        collapse = TimelineCollapse.collapse(results, fork_id=tl.fork_id)

        # Confirm original state was not touched
        assert state.world.cells[0][0].cost != 0.9
        assert len(ledger.entries()) == 0

        # Winner is deterministic
        assert collapse.winning_branch_id in [r.branch_id for r in results]

    def test_full_pipeline_three_branches_winner_is_max_yield(self) -> None:
        state = _make_state()
        ledger = _make_ledger()

        yields = [1.0, 5.0, 3.0]

        def executor(bid: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            idx = len(l.entries())  # use number of existing entries as branch index
            y = yields[idx % len(yields)]
            l.record(_make_route_result())
            return BranchResult(bid, f"b{idx}", asset=y, cost=0.5, risk=0.5, success=True)

        tl = ParallelTimeline(state, ledger)
        tl.add_branch("slow")
        tl.add_branch("fast")
        tl.add_branch("medium")
        results = tl.run(executor)
        collapse = TimelineCollapse.collapse(results, fork_id=tl.fork_id)

        assert collapse.winning_yield == max(r.yield_score() for r in results)

    def test_adversarial_executor_writes_to_arbitrary_cell(self) -> None:
        """Executor tries to overwrite every cell; source must be intact."""
        state = _make_state(size=3)
        ledger = _make_ledger()

        original_cells = [
            list(row) for row in state.world.cells
        ]

        def evil_executor(bid: str, s: GridState, l: AgentEconomyLedger) -> BranchResult:
            for r in range(s.world.size):
                for c in range(s.world.size):
                    s.world.set_cell(r, c, 0.999, 0.999, 0.001, 0.0)
            return BranchResult(bid, "evil", asset=99.0, cost=0.01, risk=0.01, success=True)

        tl = ParallelTimeline(state, ledger)
        for _ in range(5):
            tl.add_branch("attacker")
        tl.run(evil_executor)

        for r in range(state.world.size):
            for c in range(state.world.size):
                assert state.world.cells[r][c] == original_cells[r][c]
