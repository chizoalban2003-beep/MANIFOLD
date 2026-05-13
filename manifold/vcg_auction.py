"""manifold/vcg_auction.py — VCG task auction for provably truthful agent assignment.

VCG (Vickrey-Clarke-Groves) auctions are the only mechanisms provably
simultaneously efficient, individually rational, and incentive-compatible
(truthful reporting is the dominant strategy).

The domain_risk_estimate values from episodic memory (PROMPT 3) are truthful
by construction — agents cannot strategically misrepresent their own historical
outcomes.

Research: SocialGenome already exhibits evolutionary equilibrium dynamics.
The VCG mechanism gives that equilibrium a formal economic foundation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.agent_registry import AgentRegistry


@dataclass
class AgentBid:
    """A single agent's bid for a task auction."""

    agent_id: str
    domain: str
    domain_risk: float   # from episode_history (truthful by construction)
    ats: float           # agent trust score (health_score)
    valuation: float     # = ats * (1 - domain_risk) for this task type


@dataclass
class VCGResult:
    """Result of a VCG auction run."""

    assignments: dict      # task_id -> agent_id
    payments: dict         # agent_id -> governance tokens owed
    social_welfare: float  # total valuation of winning assignments
    bids: list             # list of AgentBid (all participants)
    mechanism: str = "VCG"


class VCGAuction:
    """Provably truthful task auction based on VCG mechanism.

    Parameters
    ----------
    registry:
        The AgentRegistry to source bids from.

    Example
    -------
    ::

        auction = VCGAuction(registry)
        result = auction.run("finance", required_capabilities=["billing"], n_tasks=1)
        winner = result.assignments["task_0"]
    """

    def __init__(self, registry: "AgentRegistry") -> None:
        self._registry = registry

    def _collect_bids(
        self,
        task_domain: str,
        required_capabilities: list,
    ) -> list[AgentBid]:
        """Collect bids from all capable active agents."""
        active = self._registry.active_agents()
        bids: list[AgentBid] = []
        for agent in active:
            # Filter by capabilities
            if required_capabilities and not all(
                c in agent.capabilities for c in required_capabilities
            ):
                continue
            ats = agent.health_score()
            domain_risk = self._registry.domain_risk_estimate(agent.agent_id, task_domain)
            valuation = ats * (1.0 - domain_risk)
            bids.append(AgentBid(
                agent_id=agent.agent_id,
                domain=task_domain,
                domain_risk=round(domain_risk, 4),
                ats=round(ats, 4),
                valuation=round(valuation, 6),
            ))
        return sorted(bids, key=lambda b: b.valuation, reverse=True)

    def run(
        self,
        task_domain: str,
        required_capabilities: list | None = None,
        n_tasks: int = 1,
    ) -> VCGResult:
        """Run the VCG auction and return assignments + payments.

        VCG payment for each winner i:
            payment_i = social_welfare_without_i - social_welfare_of_others_with_i

        This equals the externality winner i imposes on others (marginal
        contribution to social welfare).

        Parameters
        ----------
        task_domain:
            Domain string for this task (used for domain_risk_estimate).
        required_capabilities:
            Optional list of capabilities that bidding agents must have.
        n_tasks:
            Number of identical tasks to assign.

        Returns
        -------
        VCGResult
        """
        caps = required_capabilities or []
        bids = self._collect_bids(task_domain, caps)

        if not bids:
            return VCGResult(
                assignments={},
                payments={},
                social_welfare=0.0,
                bids=[],
            )

        # Optimal allocation: top n_tasks bidders win
        n_win = min(n_tasks, len(bids))
        winners = bids[:n_win]
        losers = bids[n_win:]

        assignments: dict[str, str] = {}
        for i, bid in enumerate(winners):
            assignments[f"task_{i}"] = bid.agent_id

        social_welfare = sum(b.valuation for b in winners)

        # VCG payments
        # payment_i = (total welfare without i assigned) - (welfare of others with i assigned)
        # For this single-unit-per-task case:
        #   Without winner i: the next best agent would win their slot.
        #   payment_i = welfare_others_without_i - welfare_others_with_i
        #   This simplifies to: payment_i = next_best_valuation (VCG = second-price)
        payments: dict[str, float] = {}
        for i, winner in enumerate(winners):
            if losers:
                # Payment = value of the best available replacement
                payment = losers[0].valuation
            elif len(winners) > 1:
                # Payment = value of the next winner (Vickrey-style)
                other_winner_vals = [w.valuation for j, w in enumerate(winners) if j != i]
                payment = max(other_winner_vals) if other_winner_vals else 0.0
            else:
                payment = 0.0  # sole bidder — no externality
            payments[winner.agent_id] = round(max(0.0, payment), 6)

        return VCGResult(
            assignments=assignments,
            payments=payments,
            social_welfare=round(social_welfare, 6),
            bids=bids,
        )
