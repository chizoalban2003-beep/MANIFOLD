"""Phase 53: Monte Carlo Scenario Simulator — Dream State Engine.

Allows MANIFOLD to "hallucinate" future failures and discover optimal
configuration before they occur in the real world.

Architecture
------------
A **simulation run** consists of *N* independent *scenarios*.  Each scenario:

1. Draws randomised fault parameters (latency, peer betrayal, tool failure
   probability, inflation factor) from configurable distributions.
2. Executes a lightweight shadow version of the
   :class:`~manifold.dag.GraphExecutor` that doesn't touch the real
   interceptor or swarm — it only runs the brain's decision logic and the
   PID threshold.
3. Records a binary ``survived`` outcome: ``True`` if the graph completed
   with the PID threshold still within bounds (no runaway entropy).

The :class:`OptimalThresholdFinder` collects all scenario outcomes and
uses a simple binning sweep to suggest the ``risk_veto_threshold`` and PID
gains (``kp``, ``ki``, ``kd``) that maximise the overall survival rate.

Key classes
-----------
``ScenarioConfig``
    Tunable parameters for the Monte Carlo run.
``FaultProfile``
    Randomised fault parameters drawn for one scenario.
``ScenarioOutcome``
    Result of simulating one scenario (survived + stats).
``SimulationReport``
    Immutable summary of a completed Monte Carlo run.
``ScenarioGenerator``
    Generates and runs scenarios; produces a :class:`SimulationReport`.
``OptimalThresholdFinder``
    Analyses a :class:`SimulationReport` and recommends optimal thresholds.
``ThresholdRecommendation``
    Recommended PID + interceptor configuration.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# ScenarioConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioConfig:
    """Tunable parameters for one Monte Carlo simulation run.

    Parameters
    ----------
    n_scenarios:
        Number of shadow scenarios to simulate.  Default: ``1_000``.
    n_nodes_per_graph:
        Number of nodes in each shadow DAG.  Default: ``5``.
    latency_max_ms:
        Upper bound for randomised per-node latency.  Default: ``2000`` ms.
    betrayal_prob_max:
        Maximum probability that any single peer will "betray" (return a
        bad outcome).  Default: ``0.3``.
    tool_failure_prob_max:
        Maximum probability of tool failure per node.  Default: ``0.4``.
    inflation_max:
        Maximum hyper-inflation multiplier applied to trust costs.
        Default: ``5.0``.
    entropy_setpoint:
        Target entropy for the shadow PID controller.  Default: ``0.3``.
    risk_veto_threshold_range:
        ``(min, max)`` sweep range for the threshold grid search.
        Default: ``(0.2, 0.8)``.
    threshold_bins:
        Number of bins in the threshold grid search.  Default: ``12``.
    seed:
        Optional RNG seed for reproducibility.
    """

    n_scenarios: int = 1_000
    n_nodes_per_graph: int = 5
    latency_max_ms: float = 2_000.0
    betrayal_prob_max: float = 0.3
    tool_failure_prob_max: float = 0.4
    inflation_max: float = 5.0
    entropy_setpoint: float = 0.3
    risk_veto_threshold_range: tuple[float, float] = (0.2, 0.8)
    threshold_bins: int = 12
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "n_scenarios": self.n_scenarios,
            "n_nodes_per_graph": self.n_nodes_per_graph,
            "latency_max_ms": self.latency_max_ms,
            "betrayal_prob_max": self.betrayal_prob_max,
            "tool_failure_prob_max": self.tool_failure_prob_max,
            "inflation_max": self.inflation_max,
            "entropy_setpoint": self.entropy_setpoint,
            "risk_veto_threshold_range": list(self.risk_veto_threshold_range),
            "threshold_bins": self.threshold_bins,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# FaultProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultProfile:
    """Randomised fault parameters drawn for one shadow scenario.

    Attributes
    ----------
    latency_ms:
        Per-node average latency in milliseconds.
    betrayal_prob:
        Probability that any peer returns a dishonest/bad result.
    tool_failure_prob:
        Per-node tool failure probability.
    inflation_factor:
        Trust cost multiplier (simulates hyper-inflation).
    initial_entropy:
        System entropy injected at the start of the scenario.
    """

    latency_ms: float
    betrayal_prob: float
    tool_failure_prob: float
    inflation_factor: float
    initial_entropy: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "latency_ms": round(self.latency_ms, 2),
            "betrayal_prob": round(self.betrayal_prob, 4),
            "tool_failure_prob": round(self.tool_failure_prob, 4),
            "inflation_factor": round(self.inflation_factor, 3),
            "initial_entropy": round(self.initial_entropy, 4),
        }


# ---------------------------------------------------------------------------
# ScenarioOutcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioOutcome:
    """Result of simulating one shadow scenario.

    Attributes
    ----------
    scenario_id:
        Zero-based index of this scenario.
    fault_profile:
        The fault parameters used.
    risk_veto_threshold:
        The interceptor threshold tested in this scenario.
    survived:
        ``True`` if the shadow graph completed without runaway entropy or
        budget exhaustion.
    nodes_succeeded:
        Number of shadow graph nodes that succeeded.
    nodes_failed:
        Number of shadow graph nodes that failed.
    final_entropy:
        System entropy at the end of the scenario.
    pid_output:
        Final PID controller output (adjusted threshold).
    """

    scenario_id: int
    fault_profile: FaultProfile
    risk_veto_threshold: float
    survived: bool
    nodes_succeeded: int
    nodes_failed: int
    final_entropy: float
    pid_output: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "scenario_id": self.scenario_id,
            "fault_profile": self.fault_profile.to_dict(),
            "risk_veto_threshold": round(self.risk_veto_threshold, 4),
            "survived": self.survived,
            "nodes_succeeded": self.nodes_succeeded,
            "nodes_failed": self.nodes_failed,
            "final_entropy": round(self.final_entropy, 4),
            "pid_output": round(self.pid_output, 4),
        }


# ---------------------------------------------------------------------------
# SimulationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationReport:
    """Immutable summary of a completed Monte Carlo run.

    Attributes
    ----------
    config:
        The :class:`ScenarioConfig` used.
    outcomes:
        Tuple of all :class:`ScenarioOutcome` objects.
    survival_rate:
        Fraction of scenarios that survived (``survived_count / n_scenarios``).
    mean_final_entropy:
        Mean final entropy across all scenarios.
    duration_seconds:
        Wall-clock duration of the simulation.
    """

    config: ScenarioConfig
    outcomes: tuple[ScenarioOutcome, ...]
    survival_rate: float
    mean_final_entropy: float
    duration_seconds: float

    @property
    def survived_count(self) -> int:
        """Number of scenarios that survived."""
        return sum(1 for o in self.outcomes if o.survived)

    @property
    def n_scenarios(self) -> int:
        """Total number of scenarios run."""
        return len(self.outcomes)

    def survival_rate_at_threshold(self, threshold: float, tol: float = 0.05) -> float:
        """Return survival rate for scenarios where threshold ≈ *threshold*.

        Parameters
        ----------
        threshold:
            Target threshold value.
        tol:
            Half-width of the tolerance band.
        """
        matching = [
            o for o in self.outcomes
            if abs(o.risk_veto_threshold - threshold) <= tol
        ]
        if not matching:
            return 0.0
        return sum(1 for o in matching if o.survived) / len(matching)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (outcomes omitted for brevity)."""
        return {
            "config": self.config.to_dict(),
            "n_scenarios": self.n_scenarios,
            "survived_count": self.survived_count,
            "survival_rate": round(self.survival_rate, 4),
            "mean_final_entropy": round(self.mean_final_entropy, 4),
            "duration_seconds": round(self.duration_seconds, 4),
        }


# ---------------------------------------------------------------------------
# Internal: shadow PID helper
# ---------------------------------------------------------------------------


def _shadow_pid_tick(
    entropy: float,
    setpoint: float,
    kp: float,
    ki: float,
    kd: float,
    prev_error: float,
    integral: float,
    dt: float = 0.1,
) -> tuple[float, float, float]:
    """One PID tick returning (output, new_integral, new_error)."""
    error = setpoint - entropy
    integral = max(-5.0, min(5.0, integral + error * dt))
    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    output = kp * error + ki * integral + kd * derivative
    return max(0.1, min(0.9, output)), integral, error


# ---------------------------------------------------------------------------
# ScenarioGenerator
# ---------------------------------------------------------------------------


@dataclass
class ScenarioGenerator:
    """Generates and runs Monte Carlo shadow scenarios.

    Parameters
    ----------
    config:
        Simulation configuration.

    Example
    -------
    ::

        gen = ScenarioGenerator(ScenarioConfig(n_scenarios=500, seed=42))
        report = gen.run()
        print(f"Survival rate: {report.survival_rate:.2%}")
    """

    config: ScenarioConfig = field(default_factory=ScenarioConfig)

    def run(self) -> SimulationReport:
        """Execute all scenarios and return a :class:`SimulationReport`.

        Returns
        -------
        SimulationReport
            Full results of the Monte Carlo run.
        """
        rng = random.Random(self.config.seed)
        t_start = time.monotonic()

        # Pre-compute threshold grid
        lo, hi = self.config.risk_veto_threshold_range
        bins = self.config.threshold_bins
        step = (hi - lo) / max(1, bins - 1)
        thresholds = [lo + i * step for i in range(bins)]

        outcomes: list[ScenarioOutcome] = []

        for i in range(self.config.n_scenarios):
            threshold = thresholds[i % len(thresholds)]
            fault = self._sample_fault(rng)
            outcome = self._simulate_scenario(i, fault, threshold, rng)
            outcomes.append(outcome)

        n = len(outcomes)
        survival_rate = sum(1 for o in outcomes if o.survived) / n if n > 0 else 0.0
        mean_entropy = (
            sum(o.final_entropy for o in outcomes) / n if n > 0 else 0.0
        )

        return SimulationReport(
            config=self.config,
            outcomes=tuple(outcomes),
            survival_rate=survival_rate,
            mean_final_entropy=mean_entropy,
            duration_seconds=time.monotonic() - t_start,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_fault(self, rng: random.Random) -> FaultProfile:
        """Draw a random fault profile."""
        return FaultProfile(
            latency_ms=rng.uniform(0.0, self.config.latency_max_ms),
            betrayal_prob=rng.uniform(0.0, self.config.betrayal_prob_max),
            tool_failure_prob=rng.uniform(0.0, self.config.tool_failure_prob_max),
            inflation_factor=rng.uniform(1.0, self.config.inflation_max),
            initial_entropy=rng.uniform(0.0, 1.0),
        )

    def _simulate_scenario(
        self,
        scenario_id: int,
        fault: FaultProfile,
        threshold: float,
        rng: random.Random,
    ) -> ScenarioOutcome:
        """Simulate one shadow DAG execution under *fault* conditions.

        The shadow graph does **not** touch any real MANIFOLD singletons.
        It only runs the decision math inline.
        """
        n_nodes = self.config.n_nodes_per_graph
        setpoint = self.config.entropy_setpoint

        # Shadow PID state
        entropy = fault.initial_entropy
        integral = 0.0
        prev_error = setpoint - entropy
        pid_threshold = threshold

        nodes_succeeded = 0
        nodes_failed = 0

        for _ in range(n_nodes):
            # Adjust entropy based on tool failure + betrayal
            if rng.random() < fault.tool_failure_prob:
                entropy = min(1.0, entropy + 0.1)
            if rng.random() < fault.betrayal_prob:
                entropy = min(1.0, entropy + 0.05)

            # PID tick
            pid_out, integral, prev_error = _shadow_pid_tick(
                entropy, setpoint, kp=1.0, ki=0.1, kd=0.05,
                prev_error=prev_error, integral=integral,
            )
            pid_threshold = pid_out

            # Shadow decision: risk = entropy * inflation_factor (capped at 1)
            risk = min(1.0, entropy * fault.inflation_factor * 0.1)
            stakes = rng.uniform(0.2, 1.0)
            if risk * stakes >= pid_threshold:
                nodes_failed += 1
                entropy = min(1.0, entropy + 0.05)
            else:
                nodes_succeeded += 1
                entropy = max(0.0, entropy - 0.02)

        # "Survived" = majority of nodes succeeded AND entropy didn't run away
        survived = (
            nodes_succeeded >= math.ceil(n_nodes / 2)
            and entropy < 0.9
        )

        return ScenarioOutcome(
            scenario_id=scenario_id,
            fault_profile=fault,
            risk_veto_threshold=threshold,
            survived=survived,
            nodes_succeeded=nodes_succeeded,
            nodes_failed=nodes_failed,
            final_entropy=entropy,
            pid_output=pid_threshold,
        )


# ---------------------------------------------------------------------------
# ThresholdRecommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Recommended PID + interceptor configuration.

    Attributes
    ----------
    risk_veto_threshold:
        The recommended ``InterceptorConfig.risk_veto_threshold``.
    suggested_kp:
        Recommended PID proportional gain.
    suggested_ki:
        Recommended PID integral gain.
    suggested_kd:
        Recommended PID derivative gain.
    best_survival_rate:
        Survival rate achieved at the recommended threshold.
    confidence:
        Fraction of scenarios that informed this recommendation (0–1).
    reasoning:
        Human-readable summary of why this threshold was chosen.
    """

    risk_veto_threshold: float
    suggested_kp: float
    suggested_ki: float
    suggested_kd: float
    best_survival_rate: float
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "risk_veto_threshold": round(self.risk_veto_threshold, 4),
            "suggested_kp": round(self.suggested_kp, 4),
            "suggested_ki": round(self.suggested_ki, 4),
            "suggested_kd": round(self.suggested_kd, 4),
            "best_survival_rate": round(self.best_survival_rate, 4),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# OptimalThresholdFinder
# ---------------------------------------------------------------------------


@dataclass
class OptimalThresholdFinder:
    """Analyses a :class:`SimulationReport` to recommend optimal thresholds.

    The finder groups scenario outcomes by their ``risk_veto_threshold`` value
    (using a configurable tolerance band), computes survival rate per bin, and
    selects the threshold with the highest survival rate.

    PID gains are adjusted heuristically based on the average final entropy
    relative to the setpoint.

    Parameters
    ----------
    tolerance:
        Half-width of the threshold-grouping tolerance band.
        Default: ``0.05``.

    Example
    -------
    ::

        finder = OptimalThresholdFinder()
        rec = finder.recommend(report)
        print(f"Optimal threshold: {rec.risk_veto_threshold:.3f} "
              f"(survival: {rec.best_survival_rate:.1%})")
    """

    tolerance: float = 0.05

    def recommend(self, report: SimulationReport) -> ThresholdRecommendation:
        """Derive a threshold recommendation from *report*.

        Parameters
        ----------
        report:
            A completed :class:`SimulationReport`.

        Returns
        -------
        ThresholdRecommendation
            The optimal configuration inferred from the Monte Carlo data.
        """
        if not report.outcomes:
            return ThresholdRecommendation(
                risk_veto_threshold=0.45,
                suggested_kp=1.0,
                suggested_ki=0.1,
                suggested_kd=0.05,
                best_survival_rate=0.0,
                confidence=0.0,
                reasoning="No scenarios available — using default configuration.",
            )

        # Collect unique threshold values from outcomes
        seen: list[float] = []
        for o in report.outcomes:
            t = o.risk_veto_threshold
            if not any(abs(t - s) < self.tolerance for s in seen):
                seen.append(t)
        seen.sort()

        # Compute survival rate per threshold bucket
        best_threshold = seen[0]
        best_rate = 0.0
        best_count = 0

        for t in seen:
            matching = [
                o for o in report.outcomes
                if abs(o.risk_veto_threshold - t) < self.tolerance
            ]
            rate = sum(1 for o in matching if o.survived) / len(matching)
            if rate > best_rate:
                best_rate = rate
                best_threshold = t
                best_count = len(matching)

        # Heuristic PID adjustment
        mean_entropy = report.mean_final_entropy
        setpoint = report.config.entropy_setpoint
        error = setpoint - mean_entropy

        # If entropy is too high → boost Kp; if too low → reduce it
        kp = max(0.5, min(3.0, 1.0 + error * 2.0))
        ki = 0.1
        kd = max(0.01, min(0.2, 0.05 - error * 0.1))

        confidence = best_count / report.n_scenarios if report.n_scenarios > 0 else 0.0

        reasoning = (
            f"Threshold {best_threshold:.3f} achieved the highest survival rate "
            f"({best_rate:.1%}) across {best_count} scenarios. "
            f"Mean entropy was {mean_entropy:.3f} vs setpoint {setpoint:.3f}; "
            f"PID gains adjusted accordingly (Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f})."
        )

        return ThresholdRecommendation(
            risk_veto_threshold=round(best_threshold, 4),
            suggested_kp=round(kp, 4),
            suggested_ki=round(ki, 4),
            suggested_kd=round(kd, 4),
            best_survival_rate=round(best_rate, 4),
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )
