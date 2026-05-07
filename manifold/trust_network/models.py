"""Phase 70: Agent Trust Score (ATS) — data models.

ToolRegistration, TrustSignal, AgentTrustScore.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

TrustTier = Literal["verified", "provisional", "flagged", "banned"]


@dataclass
class ToolRegistration:
    """A tool registered in the ATS network."""

    tool_id: str
    org_id: str
    display_name: str
    domain: str
    registered_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class TrustSignal:
    """One anonymised trust signal submitted by a MANIFOLD deployment."""

    tool_id: str
    signal_type: Literal["success", "failure", "adversarial", "escalation"]
    domain: str
    stakes: float
    submitter_hash: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentTrustScore:
    """The computed ATS for a tool, aggregated across all submitters."""

    tool_id: str
    score: float
    tier: TrustTier
    total_signals: int
    success_rate: float
    adversarial_rate: float
    last_updated: float = field(default_factory=time.time)
    contributing_orgs: int = 0

    @classmethod
    def from_signals(
        cls, tool_id: str, signals: list["TrustSignal"]
    ) -> "AgentTrustScore":
        """Compute an AgentTrustScore from a list of TrustSignals."""
        if not signals:
            return cls(
                tool_id=tool_id,
                score=0.5,
                tier="provisional",
                total_signals=0,
                success_rate=0.5,
                adversarial_rate=0.0,
            )

        total = len(signals)
        successes = sum(1 for s in signals if s.signal_type == "success")
        adversarials = sum(1 for s in signals if s.signal_type == "adversarial")
        success_rate = successes / total
        adversarial_rate = adversarials / total

        total_weight = sum(s.stakes for s in signals)
        if total_weight > 0:
            weighted_score = sum(
                (1.0 if s.signal_type == "success" else 0.0) * s.stakes
                for s in signals
            ) / total_weight
        else:
            weighted_score = 0.5

        tier: TrustTier
        if adversarial_rate > 0.15:
            tier = "banned"
        elif adversarial_rate > 0.05 or success_rate < 0.5:
            tier = "flagged"
        elif total < 20:
            tier = "provisional"
        else:
            tier = "verified"

        orgs = len({s.submitter_hash for s in signals})
        return cls(
            tool_id=tool_id,
            score=round(weighted_score, 4),
            tier=tier,
            total_signals=total,
            success_rate=round(success_rate, 4),
            adversarial_rate=round(adversarial_rate, 4),
            contributing_orgs=orgs,
        )
