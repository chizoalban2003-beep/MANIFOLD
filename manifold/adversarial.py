"""Phase 11: Adversarial Robustness & Game Theory.

Once trust is priced, rational actors (tool providers, malicious agents)
will attempt to "game" MANIFOLD's reputation system.  This module
implements three detection mechanisms that protect the ecosystem:

1. **ReputationLaunderingDetector** — flags a ``GossipNote`` source
   (Scout) that consistently inflates reputation for a specific tool
   without corresponding direct-use evidence.

2. **AdversarialPricingDetector** — detects "honey-pot" behaviour: a
   tool that performs well during the warm-up window to earn "Veteran"
   status, then reverts to failing.  Implemented as a change-point
   detector over a rolling window of outcomes.

3. **NashEquilibriumGate** — asks "is this tool's current reputation
   statistically plausible given the ambient environment noise?"  Uses
   a z-score test against a bootstrapped null distribution.  If the
   reputation is implausibly high, the gate triggers an automatic
   ``AuditTrigger`` that can deploy a Predatory Scout.

Key classes
-----------
``ToolOutcomeWindow``
    Sliding-window buffer of recent per-tool outcomes.
``ReputationLaunderingDetector``
    Gossip-source analysis: detects artificial boosting by a Scout.
``AdversarialPricingDetector``
    Change-point detector for honey-pot behaviour.
``AuditTrigger``
    Result of the ``NashEquilibriumGate``: includes trigger reason and
    recommended audit action.
``NashEquilibriumGate``
    Orchestrates all three detectors; returns ``AuditTrigger`` or None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .brain import BrainMemory, GossipNote


# ---------------------------------------------------------------------------
# ToolOutcomeWindow
# ---------------------------------------------------------------------------


@dataclass
class ToolOutcomeWindow:
    """Sliding-window buffer of recent binary outcomes for a single tool.

    Parameters
    ----------
    window_size:
        Maximum number of recent outcomes to retain.  Default: 50.

    Example
    -------
    ::

        window = ToolOutcomeWindow(window_size=20)
        for outcome in outcomes:
            window.record(outcome)
        print(window.recent_success_rate())
    """

    window_size: int = 50

    _outcomes: list[bool] = field(default_factory=list, init=False, repr=False)

    def record(self, success: bool) -> None:
        """Record one outcome; oldest entry is dropped when window is full."""
        self._outcomes.append(success)
        if len(self._outcomes) > self.window_size:
            self._outcomes.pop(0)

    def recent_success_rate(self, last_n: int | None = None) -> float:
        """Return the success rate over the most recent *last_n* outcomes.

        Parameters
        ----------
        last_n:
            Number of most recent outcomes to use.  If ``None``, uses all
            retained outcomes.

        Returns
        -------
        float
            Success rate in [0, 1].  Returns 1.0 if no outcomes recorded.
        """
        outcomes = self._outcomes[-last_n:] if last_n else self._outcomes
        if not outcomes:
            return 1.0
        return sum(outcomes) / len(outcomes)

    def count(self) -> int:
        """Return the number of outcomes currently in the window."""
        return len(self._outcomes)

    def clear(self) -> None:
        """Clear all recorded outcomes."""
        self._outcomes.clear()


# ---------------------------------------------------------------------------
# ReputationLaunderingDetector
# ---------------------------------------------------------------------------


@dataclass
class ReputationLaunderingDetector:
    """Detects Scouts that consistently boost a specific tool artificially.

    A "Reputation Launderer" is a gossip source that:
    - Sends predominantly ``"healthy"`` signals about a specific tool, AND
    - Has low or no *direct* usage of that tool (no first-hand evidence).

    Parameters
    ----------
    boost_threshold:
        Fraction of "healthy" claims that triggers a laundering alert.
        Default: 0.85 (85% of all claims about a tool are "healthy").
    min_claims:
        Minimum number of claims about a single tool before alerting.
        Default: 10.

    Example
    -------
    ::

        detector = ReputationLaunderingDetector()
        for note in gossip_notes:
            detector.record(note)
        suspects = detector.suspects()
    """

    boost_threshold: float = 0.85
    min_claims: float = 10.0

    # {source_id: {tool_name: {"healthy": int, "failing": int, "degraded": int}}}
    _claim_log: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def record(self, note: GossipNote) -> None:
        """Record a gossip note into the claim log.

        Parameters
        ----------
        note:
            A ``GossipNote`` from any source.
        """
        tool_claims = self._claim_log.setdefault(note.source_id, {}).setdefault(
            note.tool, {"healthy": 0, "failing": 0, "degraded": 0}
        )
        key = note.claim if note.claim in tool_claims else "degraded"
        tool_claims[key] += 1

    def boost_fraction(self, source_id: str, tool_name: str) -> float:
        """Return the fraction of "healthy" claims for *source_id* about *tool_name*.

        Returns 0.0 if no data.

        Parameters
        ----------
        source_id:
            The gossip source to inspect.
        tool_name:
            The tool to inspect.
        """
        claims = self._claim_log.get(source_id, {}).get(tool_name, {})
        total = sum(claims.values())
        if total == 0:
            return 0.0
        return claims.get("healthy", 0) / total

    def total_claims(self, source_id: str, tool_name: str) -> int:
        """Return total claims from *source_id* about *tool_name*."""
        claims = self._claim_log.get(source_id, {}).get(tool_name, {})
        return sum(claims.values())

    def suspects(self) -> list[dict[str, object]]:
        """Return a list of suspected launderers.

        Each entry is a dict with keys: ``source_id``, ``tool_name``,
        ``boost_fraction``, ``total_claims``.

        Returns
        -------
        list[dict]
            Suspects sorted by boost_fraction descending.
        """
        results: list[dict[str, object]] = []
        for source_id, tools in self._claim_log.items():
            for tool_name in tools:
                total = self.total_claims(source_id, tool_name)
                if total < self.min_claims:
                    continue
                bf = self.boost_fraction(source_id, tool_name)
                if bf >= self.boost_threshold:
                    results.append(
                        {
                            "source_id": source_id,
                            "tool_name": tool_name,
                            "boost_fraction": bf,
                            "total_claims": total,
                        }
                    )
        return sorted(results, key=lambda x: x["boost_fraction"], reverse=True)

    def is_suspect(self, source_id: str, tool_name: str) -> bool:
        """Return ``True`` if this source/tool pair exceeds the boost threshold."""
        if self.total_claims(source_id, tool_name) < self.min_claims:
            return False
        return self.boost_fraction(source_id, tool_name) >= self.boost_threshold


# ---------------------------------------------------------------------------
# AdversarialPricingDetector
# ---------------------------------------------------------------------------


@dataclass
class AdversarialPricingDetector:
    """Detects "honey-pot" tools that perform well early, then fail.

    The detector maintains two rolling windows:
    - **warm_up_window**: the first *warm_up_size* outcomes (the "lure" period).
    - **post_window**: outcomes after the warm-up (the "trap" period).

    A honey-pot is suspected when:
        ``warm_up_rate - post_rate >= drop_threshold``

    Parameters
    ----------
    warm_up_size:
        Number of initial outcomes constituting the warm-up window.
        Default: 20.
    post_window_size:
        Rolling window size for post-warm-up outcomes.  Default: 30.
    drop_threshold:
        Minimum success-rate drop from warm-up to post period to flag.
        Default: 0.35.
    min_post_outcomes:
        Minimum post-warm-up outcomes needed before alerting.
        Default: 10.

    Example
    -------
    ::

        detector = AdversarialPricingDetector()
        for success in [True]*20 + [False]*15:
            detector.record("gpt-4o", success)
        flag = detector.is_suspect("gpt-4o")
    """

    warm_up_size: int = 20
    post_window_size: int = 30
    drop_threshold: float = 0.35
    min_post_outcomes: int = 10

    # {tool_name: {"warm_up": [bool...], "post": ToolOutcomeWindow}}
    _state: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )

    def record(self, tool_name: str, success: bool) -> None:
        """Record one outcome for *tool_name*.

        Parameters
        ----------
        tool_name:
            The tool being tracked.
        success:
            Whether the outcome was successful.
        """
        state = self._state.setdefault(
            tool_name,
            {"warm_up": [], "post": ToolOutcomeWindow(self.post_window_size)},
        )
        warm_up: list[bool] = state["warm_up"]  # type: ignore[assignment]
        post: ToolOutcomeWindow = state["post"]  # type: ignore[assignment]
        if len(warm_up) < self.warm_up_size:
            warm_up.append(success)
        else:
            post.record(success)

    def warm_up_rate(self, tool_name: str) -> float | None:
        """Return warm-up success rate, or ``None`` if warm-up not complete."""
        state = self._state.get(tool_name)
        if state is None:
            return None
        warm_up: list[bool] = state["warm_up"]  # type: ignore[assignment]
        if len(warm_up) < self.warm_up_size:
            return None
        return sum(warm_up) / len(warm_up)

    def post_rate(self, tool_name: str) -> float | None:
        """Return post-warm-up success rate, or ``None`` if insufficient data."""
        state = self._state.get(tool_name)
        if state is None:
            return None
        post: ToolOutcomeWindow = state["post"]  # type: ignore[assignment]
        if post.count() < self.min_post_outcomes:
            return None
        return post.recent_success_rate()

    def drop(self, tool_name: str) -> float | None:
        """Return warm_up_rate - post_rate, or ``None`` if insufficient data."""
        wu = self.warm_up_rate(tool_name)
        po = self.post_rate(tool_name)
        if wu is None or po is None:
            return None
        return wu - po

    def is_suspect(self, tool_name: str) -> bool:
        """Return ``True`` if the tool shows honey-pot behaviour."""
        d = self.drop(tool_name)
        return d is not None and d >= self.drop_threshold

    def suspects(self) -> list[dict[str, object]]:
        """Return all suspected honey-pot tools.

        Each entry has keys: ``tool_name``, ``warm_up_rate``, ``post_rate``,
        ``drop``.

        Returns
        -------
        list[dict]
            Suspects sorted by drop descending.
        """
        results: list[dict[str, object]] = []
        for tool_name in self._state:
            d = self.drop(tool_name)
            wu = self.warm_up_rate(tool_name)
            po = self.post_rate(tool_name)
            if d is not None and d >= self.drop_threshold:
                results.append(
                    {
                        "tool_name": tool_name,
                        "warm_up_rate": wu,
                        "post_rate": po,
                        "drop": d,
                    }
                )
        return sorted(results, key=lambda x: x["drop"], reverse=True)


# ---------------------------------------------------------------------------
# AuditTrigger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditTrigger:
    """Result of ``NashEquilibriumGate.check()``.

    An ``AuditTrigger`` indicates that a tool's reputation is statistically
    implausible and an audit Scout should be deployed.

    Attributes
    ----------
    tool_name:
        The tool under suspicion.
    reason:
        Short description of why the audit was triggered.
    trigger_type:
        ``"laundering"`` — reputation inflation by gossip sources.
        ``"honeypot"`` — adversarial warm-up followed by failure.
        ``"implausible_rep"`` — reputation z-score exceeds the gate threshold.
    z_score:
        Z-score of the tool's success rate vs. ecosystem mean (if applicable).
    recommended_action:
        String describing the recommended next step.
    """

    tool_name: str
    reason: str
    trigger_type: Literal["laundering", "honeypot", "implausible_rep"]
    z_score: float = 0.0
    recommended_action: str = "deploy_audit_scout"


# ---------------------------------------------------------------------------
# NashEquilibriumGate
# ---------------------------------------------------------------------------


@dataclass
class NashEquilibriumGate:
    """Orchestrates all adversarial detectors and produces ``AuditTrigger`` alerts.

    The gate asks: "Is this tool's current reputation statistically plausible
    given the current ecosystem noise?"  It uses three tests:

    1. **Laundering check**: ``ReputationLaunderingDetector.is_suspect()``.
    2. **Honey-pot check**: ``AdversarialPricingDetector.is_suspect()``.
    3. **Z-score plausibility**: tool's success rate is more than
       ``zscore_threshold`` standard deviations above the ecosystem mean.

    Parameters
    ----------
    zscore_threshold:
        Number of standard deviations above ecosystem mean that constitutes
        an implausibly high reputation.  Default: 2.0.
    laundering_detector:
        Optional custom ``ReputationLaunderingDetector``.  One is created
        with defaults if not provided.
    pricing_detector:
        Optional custom ``AdversarialPricingDetector``.  One is created
        with defaults if not provided.

    Example
    -------
    ::

        gate = NashEquilibriumGate()
        for note in gossip_stream:
            gate.laundering_detector.record(note)
        for tool, success in outcome_stream:
            gate.pricing_detector.record(tool, success)

        triggers = gate.check_all(memory)
        for trigger in triggers:
            print(trigger.tool_name, trigger.trigger_type)
    """

    zscore_threshold: float = 2.0
    laundering_detector: ReputationLaunderingDetector = field(
        default_factory=ReputationLaunderingDetector
    )
    pricing_detector: AdversarialPricingDetector = field(
        default_factory=AdversarialPricingDetector
    )

    def _ecosystem_stats(self, memory: BrainMemory) -> tuple[float, float]:
        """Return (mean, std) of tool success rates in *memory*.

        Returns (1.0, 0.0) if fewer than 2 tools have data.
        """
        rates = [
            stats.get("success_rate", 1.0)
            for stats in memory.tool_stats.values()
        ]
        if len(rates) < 2:
            return 1.0, 0.0
        mean = sum(rates) / len(rates)
        variance = sum((r - mean) ** 2 for r in rates) / len(rates)
        std = math.sqrt(variance) if variance > 0 else 1e-6
        return mean, std

    def zscore(self, tool_name: str, memory: BrainMemory) -> float | None:
        """Return the z-score of *tool_name*'s success rate vs. ecosystem mean.

        Returns ``None`` if the tool is not in memory.

        Parameters
        ----------
        tool_name:
            The tool to check.
        memory:
            The ``BrainMemory`` containing tool stats.
        """
        stats = memory.tool_stats.get(tool_name)
        if stats is None:
            return None
        mean, std = self._ecosystem_stats(memory)
        if std == 0:
            return 0.0
        return (stats.get("success_rate", 1.0) - mean) / std

    def check(
        self,
        tool_name: str,
        memory: BrainMemory,
        source_id: str | None = None,
    ) -> AuditTrigger | None:
        """Run all checks for *tool_name* and return an ``AuditTrigger`` or None.

        Checks are run in priority order:
        1. Laundering (if source_id provided).
        2. Honey-pot.
        3. Z-score implausibility.

        Parameters
        ----------
        tool_name:
            The tool to check.
        memory:
            Current ``BrainMemory``.
        source_id:
            Optional gossip source ID to check for laundering.

        Returns
        -------
        AuditTrigger | None
            An audit trigger if a check fires, else ``None``.
        """
        # 1. Laundering
        if source_id and self.laundering_detector.is_suspect(source_id, tool_name):
            bf = self.laundering_detector.boost_fraction(source_id, tool_name)
            return AuditTrigger(
                tool_name=tool_name,
                reason=(
                    f"Source {source_id!r} shows {bf:.0%} healthy boost fraction "
                    f"(threshold={self.laundering_detector.boost_threshold:.0%})"
                ),
                trigger_type="laundering",
                recommended_action="discount_source_and_deploy_audit_scout",
            )

        # 2. Honey-pot
        if self.pricing_detector.is_suspect(tool_name):
            drop = self.pricing_detector.drop(tool_name)
            return AuditTrigger(
                tool_name=tool_name,
                reason=(
                    f"Success rate dropped {drop:.0%} after warm-up "
                    f"(threshold={self.pricing_detector.drop_threshold:.0%})"
                ),
                trigger_type="honeypot",
                recommended_action="flag_tool_and_deploy_audit_scout",
            )

        # 3. Z-score plausibility
        z = self.zscore(tool_name, memory)
        if z is not None and z > self.zscore_threshold:
            return AuditTrigger(
                tool_name=tool_name,
                reason=(
                    f"Z-score {z:.2f} exceeds threshold {self.zscore_threshold:.2f}; "
                    f"reputation statistically implausible"
                ),
                trigger_type="implausible_rep",
                z_score=z,
                recommended_action="deploy_audit_scout",
            )

        return None

    def check_all(
        self,
        memory: BrainMemory,
    ) -> list[AuditTrigger]:
        """Run checks for all tools present in *memory*.

        Parameters
        ----------
        memory:
            The ``BrainMemory`` to scan.

        Returns
        -------
        list[AuditTrigger]
            All triggered audits (may be empty).
        """
        triggers: list[AuditTrigger] = []
        for tool_name in memory.tool_stats:
            trigger = self.check(tool_name, memory)
            if trigger:
                triggers.append(trigger)
        return triggers
