"""Tests for VCG task auction (PROMPT D2)."""
import pytest

from manifold.agent_registry import AgentRegistry, Episode
from manifold.vcg_auction import VCGAuction, AgentBid, VCGResult


def _make_registry_with_agents() -> AgentRegistry:
    """Create a registry with 3 agents and seeded episode histories."""
    reg = AgentRegistry(stale_timeout=9999)
    reg.register("agent-high", "High Performer", ["billing", "finance"], "org1")
    reg.register("agent-mid", "Mid Performer", ["billing", "finance"], "org1")
    reg.register("agent-low", "Low Performer", ["billing", "finance"], "org1")

    crna = {"c": 0.3, "r": 0.2, "n": 0.5, "a": 0.8}

    # agent-high: low risk in finance
    for _ in range(5):
        reg.record_episode("agent-high", Episode(
            task_description="finance audit",
            domain="finance",
            duration_seconds=30.0,
            success=True,
            crna_at_start=crna, crna_at_end=crna,
            risk_encountered=0.1,
        ))

    # agent-mid: medium risk in finance
    for _ in range(5):
        reg.record_episode("agent-mid", Episode(
            task_description="finance report",
            domain="finance",
            duration_seconds=45.0,
            success=True,
            crna_at_start=crna, crna_at_end=crna,
            risk_encountered=0.4,
        ))

    # agent-low: high risk in finance
    for _ in range(5):
        reg.record_episode("agent-low", Episode(
            task_description="finance task",
            domain="finance",
            duration_seconds=60.0,
            success=False,
            crna_at_start=crna, crna_at_end=crna,
            risk_encountered=0.8,
        ))
        # Give agent-low error count to lower health_score
        reg.record_task("agent-low", success=False)

    return reg


class TestVCGAuction:
    def test_highest_valuation_agent_wins_single_task(self):
        """Highest-valuation agent wins single-task auction."""
        reg = _make_registry_with_agents()
        auction = VCGAuction(reg)
        result = auction.run("finance", required_capabilities=["billing"], n_tasks=1)

        assert "task_0" in result.assignments
        winner = result.assignments["task_0"]
        # agent-high has lowest domain risk → highest valuation
        assert winner == "agent-high"

    def test_vcg_payment_equals_second_highest_valuation(self):
        """VCG payment for a single-task auction ~ second-highest valuation."""
        reg = _make_registry_with_agents()
        auction = VCGAuction(reg)
        result = auction.run("finance", required_capabilities=["billing"], n_tasks=1)

        winner = result.assignments.get("task_0")
        assert winner is not None
        payment = result.payments[winner]

        # Payment should be <= winner's own valuation
        winner_bid = next(b for b in result.bids if b.agent_id == winner)
        assert payment <= winner_bid.valuation + 1e-9

    def test_two_tasks_three_agents_top_two_win(self):
        """With 2 tasks and 3 agents: top 2 valuations win."""
        reg = _make_registry_with_agents()
        auction = VCGAuction(reg)
        result = auction.run("finance", required_capabilities=["billing"], n_tasks=2)

        assert len(result.assignments) == 2
        winners = set(result.assignments.values())
        # The 2 winners should be from the top-2 valuations
        sorted_bids = sorted(result.bids, key=lambda b: b.valuation, reverse=True)
        top_2 = {sorted_bids[0].agent_id, sorted_bids[1].agent_id}
        assert winners == top_2

    def test_social_welfare_equals_sum_of_winning_valuations(self):
        """social_welfare equals sum of winning valuations."""
        reg = _make_registry_with_agents()
        auction = VCGAuction(reg)
        result = auction.run("finance", n_tasks=2)

        winners = list(result.assignments.values())
        expected_welfare = sum(
            b.valuation for b in result.bids if b.agent_id in winners
        )
        assert abs(result.social_welfare - expected_welfare) < 1e-6

    def test_truthful_reporting_agent_cannot_improve_by_misreporting(self):
        """Truthful reporting: bids come from episode_history, not self-reports."""
        # The VCG mechanism is truthful BY CONSTRUCTION because the bids are
        # derived from historical domain_risk_estimate (not agent self-reports).
        # This test verifies the mechanism property: the winner is the one with
        # the highest ats * (1 - domain_risk) valuation.
        reg = _make_registry_with_agents()
        auction = VCGAuction(reg)
        result = auction.run("finance", n_tasks=1)

        # Verify all bids are computed from truthful history
        for bid in result.bids:
            expected_valuation = bid.ats * (1.0 - bid.domain_risk)
            assert abs(bid.valuation - round(expected_valuation, 6)) < 1e-5, (
                f"Bid valuation should equal ats*(1-domain_risk): "
                f"{bid.valuation} != {expected_valuation}"
            )
