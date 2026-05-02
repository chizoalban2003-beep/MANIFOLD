"""Phase 12: Automated Rule Discovery — the "Law-Maker".

Currently, MANIFOLD rules are added manually:
``world.add_rule("late_delivery", penalty=8.2, triggers="miss_target")``.

Phase 12 moves rule management into **Auto-Discovery**:

1. **PenaltyOptimizer** — observes ``AssetAdapter`` delta signals.  If a
   rule trigger consistently correlates with large negative asset deltas,
   the optimizer proposes an updated penalty that aligns with the realized
   loss, or adjusts it automatically if ``auto_adjust=True``.

2. **PolicySynthesizer** — inspects logs of ``HierarchicalDecision`` objects
   (successful decompositions) and extracts recurring structural patterns as
   "Best Practice" templates for specific domains.

3. **AutoRuleDiscovery** — coordinator that wraps both components and
   provides a single ``observe`` / ``suggest`` interface.

Key classes
-----------
``RuleObservation``
    A single observation linking a trigger event to an asset delta.
``PenaltyProposal``
    The output of ``PenaltyOptimizer.suggest()``: recommended penalty change.
``DecompositionTemplate``
    A "Best Practice" template produced by ``PolicySynthesizer``.
``PenaltyOptimizer``
    Maps trigger → observed asset loss; proposes aligned penalties.
``PolicySynthesizer``
    Extracts recurring decomposition patterns from decision logs.
``AutoRuleDiscovery``
    Coordinator: observe events and produce actionable rule updates.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

from .brain import BrainOutcome, BrainTask, HierarchicalDecision, SubTaskSpec
from .gridmapper import Rule
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# RuleObservation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleObservation:
    """A single observation linking a rule trigger to a realised asset delta.

    Attributes
    ----------
    trigger:
        The rule trigger string that fired (e.g. ``"miss_target"``).
    rule_name:
        Human-readable rule name (e.g. ``"late_delivery"``).
    observed_asset_delta:
        Realised change in asset value due to this event.  Negative = loss.
    current_penalty:
        The penalty currently set for this rule.
    """

    trigger: str
    rule_name: str
    observed_asset_delta: float
    current_penalty: float


# ---------------------------------------------------------------------------
# PenaltyProposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PenaltyProposal:
    """Recommended penalty adjustment produced by ``PenaltyOptimizer``.

    Attributes
    ----------
    rule_name:
        The rule being updated.
    trigger:
        The trigger string.
    current_penalty:
        The penalty in effect before the proposal.
    proposed_penalty:
        The calibrated penalty that aligns with observed loss.
    delta:
        ``proposed_penalty - current_penalty``.
    n_observations:
        Number of observations used to calibrate.
    confidence:
        Calibration confidence [0, 1]; higher with more observations.
    rationale:
        Short human-readable explanation.
    """

    rule_name: str
    trigger: str
    current_penalty: float
    proposed_penalty: float
    delta: float
    n_observations: int
    confidence: float
    rationale: str


# ---------------------------------------------------------------------------
# PenaltyOptimizer
# ---------------------------------------------------------------------------


@dataclass
class PenaltyOptimizer:
    """Aligns rule penalties with observed ``AssetAdapter`` loss signals.

    The optimizer accumulates ``RuleObservation`` records.  When at least
    ``min_observations`` have been collected for a trigger, it computes a
    calibrated penalty:

        ``calibrated = |mean(observed_asset_delta)| * penalty_scale``

    If ``auto_adjust=True``, the optimizer directly mutates the ``Rule``
    objects in ``rules_registry`` when a proposal is accepted (delta ≥
    ``adjustment_threshold``).

    Parameters
    ----------
    min_observations:
        Minimum observations per trigger before a proposal is generated.
        Default: 5.
    penalty_scale:
        Multiplier applied to the observed mean loss magnitude when computing
        the calibrated penalty.  A value of 1.0 = "charge exactly the observed
        loss"; values > 1.0 make the system more conservative.  Default: 1.2.
    adjustment_threshold:
        Absolute delta between current and proposed penalty required before
        the optimizer flags the rule as needing adjustment.  Default: 0.5.
    auto_adjust:
        If ``True``, automatically update ``Rule.penalty`` in
        ``rules_registry`` when a proposal fires.  Default: ``False``.

    Example
    -------
    ::

        optimizer = PenaltyOptimizer(min_observations=5, auto_adjust=True)
        for obs in rule_observations:
            optimizer.record(obs)

        proposals = optimizer.suggest_all()
        for p in proposals:
            print(p.rule_name, p.current_penalty, "→", p.proposed_penalty)
    """

    min_observations: int = 5
    penalty_scale: float = 1.2
    adjustment_threshold: float = 0.5
    auto_adjust: bool = False

    rules_registry: dict[str, Rule] = field(default_factory=dict)

    _observations: dict[str, list[RuleObservation]] = field(
        default_factory=dict, init=False, repr=False
    )

    def record(self, observation: RuleObservation) -> None:
        """Record a rule trigger observation.

        Parameters
        ----------
        observation:
            The observation to log.
        """
        self._observations.setdefault(observation.trigger, []).append(observation)

    def record_from_outcome(
        self,
        rule_name: str,
        trigger: str,
        outcome: BrainOutcome,
        current_penalty: float,
    ) -> None:
        """Convenience: build and record an observation from a ``BrainOutcome``.

        Parameters
        ----------
        rule_name:
            The rule name.
        trigger:
            The trigger string.
        outcome:
            The ``BrainOutcome`` that fired this rule.
        current_penalty:
            The current penalty for this rule.
        """
        delta = outcome.utility if outcome.success else -abs(current_penalty * 0.5)
        obs = RuleObservation(
            trigger=trigger,
            rule_name=rule_name,
            observed_asset_delta=delta,
            current_penalty=current_penalty,
        )
        self.record(obs)

    def suggest(self, trigger: str) -> PenaltyProposal | None:
        """Produce a penalty proposal for *trigger* if enough data exists.

        Returns ``None`` if fewer than ``min_observations`` have been
        collected for this trigger.

        Parameters
        ----------
        trigger:
            The rule trigger to calibrate.

        Returns
        -------
        PenaltyProposal | None
        """
        obs_list = self._observations.get(trigger, [])
        if len(obs_list) < self.min_observations:
            return None

        deltas = [o.observed_asset_delta for o in obs_list]
        mean_loss = abs(statistics.mean(deltas))
        current_penalty = obs_list[-1].current_penalty
        rule_name = obs_list[-1].rule_name
        proposed = round(mean_loss * self.penalty_scale, 4)
        delta = proposed - current_penalty
        n = len(obs_list)
        confidence = clamp01(min(1.0, n / (self.min_observations * 4)))

        if abs(delta) < self.adjustment_threshold:
            rationale = (
                f"Observed mean |loss| = {mean_loss:.3f}; penalty {current_penalty:.3f} "
                f"is within tolerance (delta={delta:+.3f}). No change needed."
            )
        else:
            direction = "increase" if delta > 0 else "decrease"
            rationale = (
                f"Observed mean |loss| = {mean_loss:.3f} over {n} events. "
                f"Recommend {direction} of penalty from {current_penalty:.3f} "
                f"to {proposed:.3f} (delta={delta:+.3f})."
            )

        proposal = PenaltyProposal(
            rule_name=rule_name,
            trigger=trigger,
            current_penalty=current_penalty,
            proposed_penalty=proposed,
            delta=delta,
            n_observations=n,
            confidence=confidence,
            rationale=rationale,
        )

        if self.auto_adjust and abs(delta) >= self.adjustment_threshold:
            rule = self.rules_registry.get(rule_name)
            if rule is not None:
                updated = Rule(
                    name=rule.name, penalty=proposed, triggers=rule.triggers
                )
                self.rules_registry[rule_name] = updated

        return proposal

    def suggest_all(self) -> list[PenaltyProposal]:
        """Return proposals for all triggers with enough data.

        Returns
        -------
        list[PenaltyProposal]
            Sorted by |delta| descending.
        """
        results: list[PenaltyProposal] = []
        for trigger in self._observations:
            p = self.suggest(trigger)
            if p is not None:
                results.append(p)
        return sorted(results, key=lambda x: abs(x.delta), reverse=True)

    def observation_count(self, trigger: str) -> int:
        """Return the number of observations for *trigger*."""
        return len(self._observations.get(trigger, []))


# ---------------------------------------------------------------------------
# DecompositionTemplate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecompositionTemplate:
    """A "Best Practice" decomposition pattern for a specific domain.

    Produced by ``PolicySynthesizer`` when a structural pattern appears
    in multiple successful ``HierarchicalDecision`` logs.

    Attributes
    ----------
    domain:
        The domain this template applies to (e.g. ``"legal"``, ``"support"``).
    sub_task_pattern:
        Ordered list of sub-task descriptors: (prompt_hint, relative_complexity,
        weight).  These are generalised from observed decompositions.
    avg_coordination_tax:
        Average coordination tax observed across successful decompositions.
    avg_combined_utility:
        Mean combined utility of decisions matching this pattern.
    occurrence_count:
        Number of successful decompositions this pattern was derived from.
    confidence:
        Confidence that this template is genuinely representative [0, 1].
    """

    domain: str
    sub_task_pattern: tuple[tuple[str, float, float], ...]
    avg_coordination_tax: float
    avg_combined_utility: float
    occurrence_count: int
    confidence: float


# ---------------------------------------------------------------------------
# PolicySynthesizer
# ---------------------------------------------------------------------------


@dataclass
class PolicySynthesizer:
    """Extracts recurring decomposition patterns from ``HierarchicalDecision`` logs.

    The synthesizer inspects all recorded successful decompositions for a
    domain and looks for structural similarity: decisions with the same
    number of sub-tasks where each sub-task has a similar complexity bucket.

    A decomposition is "successful" if ``combined_utility > min_utility``.
    A pattern is "recurring" if it appears ``min_occurrences`` or more times.

    Parameters
    ----------
    min_utility:
        Minimum ``combined_utility`` for a decomposition to be logged.
        Default: 0.0.
    min_occurrences:
        Minimum number of times a pattern must appear to be synthesised.
        Default: 3.
    complexity_bucket_size:
        Complexity values are bucketed at this granularity for pattern
        matching.  Default: 0.2.

    Example
    -------
    ::

        synth = PolicySynthesizer()
        for decision in successful_decompositions:
            synth.record(decision, domain="legal")

        templates = synth.synthesise("legal")
        for t in templates:
            print(t.domain, t.sub_task_pattern)
    """

    min_utility: float = 0.0
    min_occurrences: int = 3
    complexity_bucket_size: float = 0.2

    _logs: dict[str, list[HierarchicalDecision]] = field(
        default_factory=dict, init=False, repr=False
    )

    def record(self, decision: HierarchicalDecision, domain: str = "general") -> None:
        """Record a ``HierarchicalDecision`` for later pattern synthesis.

        Only decomposed decisions above ``min_utility`` are retained.

        Parameters
        ----------
        decision:
            The decision to log.
        domain:
            The domain this decision was made in.
        """
        if not decision.decomposed:
            return
        if decision.combined_utility < self.min_utility:
            return
        self._logs.setdefault(domain, []).append(decision)

    def _complexity_bucket(self, complexity: float) -> float:
        """Round *complexity* to the nearest bucket boundary."""
        b = self.complexity_bucket_size
        return round(round(complexity / b) * b, 4)

    def _decision_fingerprint(
        self, decision: HierarchicalDecision
    ) -> tuple[tuple[float, float], ...] | None:
        """Return a fingerprint tuple for pattern matching.

        Returns ``None`` if the decision has no plan or sub-tasks.
        """
        if decision.plan is None or not decision.plan.sub_tasks:
            return None
        return tuple(
            (self._complexity_bucket(st.complexity), round(st.weight, 2))
            for st in decision.plan.sub_tasks
        )

    def synthesise(self, domain: str) -> list[DecompositionTemplate]:
        """Synthesise best-practice templates for *domain*.

        Returns an empty list if no decisions have been logged or no
        pattern meets ``min_occurrences``.

        Parameters
        ----------
        domain:
            The domain to synthesise for.

        Returns
        -------
        list[DecompositionTemplate]
            Templates sorted by occurrence_count descending.
        """
        decisions = self._logs.get(domain, [])
        if not decisions:
            return []

        # Group by fingerprint
        pattern_groups: dict[
            tuple[tuple[float, float], ...], list[HierarchicalDecision]
        ] = {}
        for dec in decisions:
            fp = self._decision_fingerprint(dec)
            if fp is None:
                continue
            pattern_groups.setdefault(fp, []).append(dec)

        templates: list[DecompositionTemplate] = []
        for fp, group in pattern_groups.items():
            if len(group) < self.min_occurrences:
                continue

            # Build averaged sub-task descriptors
            plan = group[0].plan
            assert plan is not None
            sub_task_pattern: list[tuple[str, float, float]] = []
            for st in plan.sub_tasks:
                avg_complexity = statistics.mean(
                    g.plan.sub_tasks[plan.sub_tasks.index(st)].complexity  # type: ignore[union-attr]
                    for g in group
                    if g.plan and len(g.plan.sub_tasks) > plan.sub_tasks.index(st)
                )
                sub_task_pattern.append(
                    (
                        st.prompt[:40] + ("…" if len(st.prompt) > 40 else ""),
                        round(avg_complexity, 3),
                        round(st.weight, 3),
                    )
                )

            avg_tax = statistics.mean(
                g.plan.coordination_tax for g in group if g.plan
            )
            avg_utility = statistics.mean(g.combined_utility for g in group)
            n = len(group)
            confidence = clamp01(min(1.0, n / (self.min_occurrences * 5)))

            templates.append(
                DecompositionTemplate(
                    domain=domain,
                    sub_task_pattern=tuple(sub_task_pattern),
                    avg_coordination_tax=round(avg_tax, 4),
                    avg_combined_utility=round(avg_utility, 4),
                    occurrence_count=n,
                    confidence=confidence,
                )
            )

        return sorted(templates, key=lambda t: t.occurrence_count, reverse=True)

    def known_domains(self) -> list[str]:
        """Return all domains with recorded decisions."""
        return list(self._logs.keys())

    def decision_count(self, domain: str) -> int:
        """Return the number of logged decisions for *domain*."""
        return len(self._logs.get(domain, []))


# ---------------------------------------------------------------------------
# AutoRuleDiscovery
# ---------------------------------------------------------------------------


@dataclass
class AutoRuleDiscovery:
    """Coordinator that wraps ``PenaltyOptimizer`` and ``PolicySynthesizer``.

    Provides a single ``observe`` / ``suggest`` interface for the full
    Phase 12 auto-discovery pipeline.

    Parameters
    ----------
    optimizer:
        ``PenaltyOptimizer`` instance.  Created with defaults if not provided.
    synthesizer:
        ``PolicySynthesizer`` instance.  Created with defaults if not provided.

    Example
    -------
    ::

        discovery = AutoRuleDiscovery()
        discovery.observe_rule_event("late_delivery", "miss_target", outcome, penalty=8.2)
        discovery.observe_decomposition(decision, domain="legal")

        penalty_proposals = discovery.suggest_penalty_updates()
        policy_templates = discovery.suggest_policy_templates("legal")
    """

    optimizer: PenaltyOptimizer = field(default_factory=PenaltyOptimizer)
    synthesizer: PolicySynthesizer = field(default_factory=PolicySynthesizer)

    def observe_rule_event(
        self,
        rule_name: str,
        trigger: str,
        outcome: BrainOutcome,
        penalty: float,
    ) -> None:
        """Record a rule-trigger event linked to a ``BrainOutcome``.

        Parameters
        ----------
        rule_name:
            The rule name.
        trigger:
            The trigger string.
        outcome:
            The outcome that triggered the rule.
        penalty:
            The current penalty for this rule.
        """
        self.optimizer.record_from_outcome(rule_name, trigger, outcome, penalty)

    def observe_decomposition(
        self, decision: HierarchicalDecision, domain: str = "general"
    ) -> None:
        """Record a hierarchical decision for policy synthesis.

        Parameters
        ----------
        decision:
            The decision to log.
        domain:
            The domain this decision was made in.
        """
        self.synthesizer.record(decision, domain=domain)

    def suggest_penalty_updates(self) -> list[PenaltyProposal]:
        """Return all pending penalty proposals from the optimizer.

        Returns
        -------
        list[PenaltyProposal]
            Sorted by |delta| descending.
        """
        return self.optimizer.suggest_all()

    def suggest_policy_templates(self, domain: str) -> list[DecompositionTemplate]:
        """Return synthesised templates for *domain*.

        Parameters
        ----------
        domain:
            The domain to synthesise for.

        Returns
        -------
        list[DecompositionTemplate]
            Sorted by occurrence_count descending.
        """
        return self.synthesizer.synthesise(domain)

    def status(self) -> dict[str, object]:
        """Return a summary of the discovery pipeline status.

        Returns
        -------
        dict
            Keys: ``known_triggers``, ``known_domains``, ``pending_proposals``.
        """
        proposals = self.suggest_penalty_updates()
        return {
            "known_triggers": list(self.optimizer._observations.keys()),
            "known_domains": self.synthesizer.known_domains(),
            "pending_proposals": len(proposals),
        }
