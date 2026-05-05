"""Phase 27: Braintrust Consensus — Level-5 Multi-Brain Decision Making.

Three ``ManifoldBrain`` instances with distinct "Social Genomes" (tuned
``BrainConfig`` + tool profiles) cast weighted votes on every task.  The
action is only approved when the weighted consensus score exceeds the
``InterceptorConfig.risk_veto_threshold``.

Social Genomes
--------------
``risk_averse``
    Prioritises safety; high predator pressure, low delegation cost.
``asset_aggressive``
    Maximises utility; low predator pressure, low exploration cost.
``balanced``
    Default settings — moderate risk/reward profile.

Key classes
-----------
``SocialGenome``
    Named bundle of ``BrainConfig`` overrides defining a brain's personality.
``ConsensusVote``
    A single brain's vote: action + confidence + weighted score.
``ConsensusResult``
    Full outcome of a Braintrust evaluation.
``Braintrust``
    Orchestrates three brains, collects weighted votes, and returns a
    ``ConsensusResult`` with the winning action and approval status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .brain import BrainConfig, BrainDecision, BrainTask, ManifoldBrain
from .interceptor import InterceptorConfig
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# SocialGenome
# ---------------------------------------------------------------------------

PersonalityName = Literal["risk_averse", "asset_aggressive", "balanced"]


@dataclass(frozen=True)
class SocialGenome:
    """Named configuration bundle for a Braintrust personality.

    Parameters
    ----------
    name:
        Unique personality identifier.
    config:
        ``BrainConfig`` tuned to this personality.
    vote_weight:
        How much this brain's vote counts in the weighted consensus [0, 1].
        Weights are normalised across the panel before scoring.
    """

    name: PersonalityName
    config: BrainConfig
    vote_weight: float = 1.0

    @classmethod
    def risk_averse(cls) -> "SocialGenome":
        """Return the risk-averse genome: safety first, minimal delegation."""
        return cls(
            name="risk_averse",
            config=BrainConfig(
                grid_size=11,
                generations=30,
                population_size=48,
                predators=0.15,        # more adversarial pressure
                seed=1111,
                planning_cost=0.05,    # cheap to plan
                exploration_cost=0.20, # expensive to explore
                delegation_cost=0.40,  # very expensive to delegate
            ),
            vote_weight=1.2,           # tie-break toward safety
        )

    @classmethod
    def asset_aggressive(cls) -> "SocialGenome":
        """Return the asset-aggressive genome: maximise utility, accept more risk."""
        return cls(
            name="asset_aggressive",
            config=BrainConfig(
                grid_size=11,
                generations=30,
                population_size=48,
                predators=0.02,        # minimal adversarial pressure
                seed=2222,
                planning_cost=0.15,    # planning is costly (bias toward action)
                exploration_cost=0.08, # cheap to explore
                delegation_cost=0.15,  # cheap to delegate
            ),
            vote_weight=1.0,
        )

    @classmethod
    def balanced(cls) -> "SocialGenome":
        """Return the balanced genome: default MANIFOLD settings."""
        return cls(
            name="balanced",
            config=BrainConfig(
                grid_size=11,
                generations=30,
                population_size=48,
                predators=0.05,
                seed=2500,
                planning_cost=0.10,
                exploration_cost=0.12,
                delegation_cost=0.25,
            ),
            vote_weight=1.0,
        )


# ---------------------------------------------------------------------------
# ConsensusVote
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusVote:
    """A single brain's vote in a Braintrust evaluation.

    Attributes
    ----------
    genome_name:
        Name of the personality that cast this vote.
    decision:
        Full ``BrainDecision`` from this brain.
    vote_weight:
        The normalised weight of this vote in the consensus.
    approves:
        ``True`` when this brain approves the action (not refuse/escalate).
    weighted_confidence:
        ``confidence * vote_weight``.
    """

    genome_name: PersonalityName
    decision: BrainDecision
    vote_weight: float
    approves: bool
    weighted_confidence: float


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusResult:
    """Full outcome of a Braintrust evaluation.

    Attributes
    ----------
    task:
        The task that was evaluated.
    votes:
        One ``ConsensusVote`` per panel member.
    consensus_score:
        Weighted fraction of approving votes [0, 1].
    threshold:
        The minimum score required for approval.
    approved:
        ``True`` when ``consensus_score >= threshold``.
    winning_action:
        The action recommended by the highest-weighted approving brain, or
        ``"refuse"`` if no brain approved.
    reason:
        Human-readable explanation.
    """

    task: BrainTask
    votes: tuple[ConsensusVote, ...]
    consensus_score: float
    threshold: float
    approved: bool
    winning_action: str
    reason: str


# ---------------------------------------------------------------------------
# Braintrust
# ---------------------------------------------------------------------------


@dataclass
class Braintrust:
    """Three-brain consensus panel for Level-5 decision making.

    Instantiates three ``ManifoldBrain`` objects with distinct Social Genomes
    and combines their votes via a weighted average.  The action is approved
    when the consensus score ≥ *threshold*.

    Parameters
    ----------
    config:
        ``InterceptorConfig`` whose ``risk_veto_threshold`` is used as the
        approval threshold.
    tools:
        Tool profiles passed to each brain.  Defaults to an empty list.
    genomes:
        Custom panel of ``SocialGenome`` objects.  Defaults to the three
        built-in personalities.

    Example
    -------
    ::

        bt = Braintrust()
        task = BrainTask(prompt="Transfer $50k", domain="finance", stakes=0.9)
        result = bt.evaluate(task)
        if result.approved:
            print("Braintrust approved:", result.winning_action)
        else:
            print("Braintrust refused:", result.reason)
    """

    config: InterceptorConfig = field(default_factory=InterceptorConfig)
    tools: list[object] = field(default_factory=list)
    genomes: list[SocialGenome] | None = None

    _panel: list[tuple[SocialGenome, ManifoldBrain]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        panel_genomes = self.genomes or [
            SocialGenome.risk_averse(),
            SocialGenome.asset_aggressive(),
            SocialGenome.balanced(),
        ]
        self._panel = [
            (g, ManifoldBrain(config=g.config, tools=list(self.tools)))
            for g in panel_genomes
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, task: BrainTask) -> ConsensusResult:
        """Run the task through all panel brains and compute weighted consensus.

        Parameters
        ----------
        task:
            The task to evaluate.

        Returns
        -------
        ConsensusResult
            Full consensus outcome including individual votes.
        """
        raw_weights = [g.vote_weight for g, _ in self._panel]
        total_weight = sum(raw_weights) or 1.0
        normalised = [w / total_weight for w in raw_weights]

        veto_actions: frozenset[str] = frozenset({"refuse", "escalate"})
        votes: list[ConsensusVote] = []
        for (genome, brain), norm_w in zip(self._panel, normalised):
            decision: BrainDecision = brain.decide(task)
            approves = decision.action not in veto_actions
            votes.append(
                ConsensusVote(
                    genome_name=genome.name,
                    decision=decision,
                    vote_weight=norm_w,
                    approves=approves,
                    weighted_confidence=clamp01(decision.confidence) * norm_w,
                )
            )

        # Weighted approval score
        approval_score = sum(v.vote_weight for v in votes if v.approves)
        threshold = clamp01(1.0 - self.config.risk_veto_threshold)
        approved = approval_score >= threshold

        # Pick the winning action from the highest-weight approving vote
        winning_action = "refuse"
        best_wc: float = -1.0
        for v in votes:
            if v.approves and v.weighted_confidence > best_wc:
                best_wc = v.weighted_confidence
                winning_action = v.decision.action

        if approved:
            reason = (
                f"consensus_score={approval_score:.3f} >= threshold={threshold:.3f}; "
                f"winning_action={winning_action!r}"
            )
        else:
            reason = (
                f"consensus_score={approval_score:.3f} < threshold={threshold:.3f}; "
                f"Braintrust vetoed action"
            )

        return ConsensusResult(
            task=task,
            votes=tuple(votes),
            consensus_score=approval_score,
            threshold=threshold,
            approved=approved,
            winning_action=winning_action,
            reason=reason,
        )

    def panel_summary(self) -> dict[str, object]:
        """Return a lightweight summary of the panel configuration."""
        return {
            "panel_size": len(self._panel),
            "genomes": [g.name for g, _ in self._panel],
            "threshold": clamp01(1.0 - self.config.risk_veto_threshold),
        }
