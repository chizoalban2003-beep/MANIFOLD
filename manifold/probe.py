"""Phase 31: Active Canary Probing — Autonomous Threat-Intelligence Gathering.

The ``ActiveProber`` periodically selects tools with rising entropy scores
(Phase 26) and fires synthetic zero-stakes ``BrainTask`` "canary" tasks at
them.  Results are evaluated by the ``AdversarialPricingDetector`` (Phase 11).
If the tool fails or is flagged as adversarial, the ``ReputationHub`` is
penalised **without** exposing real user data to the degraded tool.

Key classes
-----------
``CanaryResult``
    Immutable record of a single probe execution.
``CanaryGenerator``
    Produces synthetic zero-stakes ``BrainTask`` objects for diagnostic use.
``ActiveProber``
    Background daemon / on-demand prober that fires canary tasks at
    high-entropy tools and updates the hub on failure.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .adversarial import AdversarialPricingDetector
from .brain import BrainTask, ManifoldBrain
from .entropy import ReputationDecay
from .federation import FederatedGossipPacket
from .hub import ReputationHub


# ---------------------------------------------------------------------------
# CanaryResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryResult:
    """Immutable record of a single canary probe execution.

    Attributes
    ----------
    tool_name:
        The tool that was probed.
    entropy_score_before:
        The entropy score that triggered the probe.
    adversarial_suspect:
        Whether the ``AdversarialPricingDetector`` flagged this tool.
    timestamp:
        POSIX timestamp when the probe was executed.
    probe_action:
        ``"pass"`` — brain approved the canary task (tool appears healthy).
        ``"fail"`` — brain refused/escalated the canary task.
        ``"suspect"`` — adversarial detection triggered.
    penalty_applied:
        ``True`` if a failing-signal gossip packet was submitted to the hub.
    """

    tool_name: str
    entropy_score_before: float
    adversarial_suspect: bool
    timestamp: float
    probe_action: str  # "pass" | "fail" | "suspect"
    penalty_applied: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "tool_name": self.tool_name,
            "entropy_score_before": self.entropy_score_before,
            "adversarial_suspect": self.adversarial_suspect,
            "timestamp": self.timestamp,
            "probe_action": self.probe_action,
            "penalty_applied": self.penalty_applied,
        }


# ---------------------------------------------------------------------------
# CanaryGenerator
# ---------------------------------------------------------------------------


@dataclass
class CanaryGenerator:
    """Generates synthetic zero-stakes ``BrainTask`` objects for canary probing.

    Canary tasks have ``stakes=0.0`` and ``safety_sensitivity=0.0`` so they
    can never trigger a real veto — they exist purely to observe the tool's
    behaviour without exposing real user data.

    Parameters
    ----------
    domain:
        Default task domain.  Each call may override this per-tool.
    """

    domain: str = "general"

    def generate(self, tool_name: str, *, domain: str | None = None) -> BrainTask:
        """Return a canary ``BrainTask`` for *tool_name*.

        Parameters
        ----------
        tool_name:
            The tool being probed (embedded in the prompt for traceability).
        domain:
            Override domain for this specific probe.

        Returns
        -------
        BrainTask
            Zero-stakes diagnostic task.
        """
        return BrainTask(
            prompt=f"canary_probe:{tool_name}",
            domain=domain or self.domain,
            stakes=0.0,
            uncertainty=0.1,
            complexity=0.1,
            safety_sensitivity=0.0,
        )


# ---------------------------------------------------------------------------
# ActiveProber
# ---------------------------------------------------------------------------


@dataclass
class ActiveProber:
    """Autonomous canary prober that fires diagnostic tasks at high-entropy tools.

    The prober selects tools whose ``entropy_score`` (from Phase 26) exceeds
    ``entropy_threshold``, fires a ``CanaryGenerator`` task, evaluates the
    brain's decision, and records an adversarial-detection outcome.  If the
    tool fails or is flagged, a ``"failing"`` gossip packet is submitted to
    the ``ReputationHub`` with a small weight — penalising the tool without
    ever routing real traffic to it.

    The prober can run in **background daemon mode** via :meth:`start` /
    :meth:`stop`, or be called on-demand via :meth:`probe_once` and
    :meth:`probe_high_entropy_tools`.

    Parameters
    ----------
    hub:
        The ``ReputationHub`` to query and penalise.
    decay:
        The ``ReputationDecay`` engine for entropy score lookups.
    brain:
        A ``ManifoldBrain`` used to evaluate canary tasks.
    entropy_threshold:
        Minimum entropy score required to trigger a probe.  Default: ``0.5``.
    interval_seconds:
        Background loop sleep interval in seconds.  Default: ``60``.
    penalty_weight:
        Gossip packet weight applied to the hub on a failing probe.
        Default: ``0.05`` (small, so a single canary doesn't dominate).

    Example
    -------
    ::

        prober = ActiveProber(hub=hub, decay=decay, brain=brain)
        results = prober.probe_high_entropy_tools()
        prober.start()   # background daemon
        # ... later ...
        prober.stop()
    """

    hub: ReputationHub
    decay: ReputationDecay
    brain: ManifoldBrain
    entropy_threshold: float = 0.5
    interval_seconds: float = 60.0
    penalty_weight: float = 0.05

    _detector: AdversarialPricingDetector = field(
        default_factory=AdversarialPricingDetector, init=False, repr=False
    )
    _generator: CanaryGenerator = field(
        default_factory=CanaryGenerator, init=False, repr=False
    )
    _results: list[CanaryResult] = field(
        default_factory=list, init=False, repr=False
    )
    _running: bool = field(default=False, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # On-demand probing
    # ------------------------------------------------------------------

    def probe_once(self, tool_name: str) -> CanaryResult:
        """Fire a single canary probe at *tool_name* and record the outcome.

        Steps:
        1. Record current entropy score.
        2. Generate a zero-stakes canary ``BrainTask``.
        3. Ask the ``ManifoldBrain`` to decide the task.
        4. Feed the result into the ``AdversarialPricingDetector``.
        5. If the probe fails or is adversarial, submit a penalty gossip packet.

        Parameters
        ----------
        tool_name:
            The tool to probe.

        Returns
        -------
        CanaryResult
            Full probe result record.
        """
        entropy = self.decay.entropy_score(tool_name)
        canary_task = self._generator.generate(tool_name)

        # Evaluate using the brain (zero-stakes so never blocks real traffic)
        decision = self.brain.decide(canary_task)
        brain_fail = decision.action in {"refuse", "escalate"}

        # Update adversarial detector
        self._detector.record(tool_name, not brain_fail)
        adversarial_suspect = self._detector.is_suspect(tool_name)

        # Determine probe outcome
        if adversarial_suspect:
            probe_action = "suspect"
        elif brain_fail:
            probe_action = "fail"
        else:
            probe_action = "pass"

        # Apply penalty to hub if probe fails
        penalty_applied = False
        if probe_action in {"fail", "suspect"}:
            penalty_packet = FederatedGossipPacket(
                tool_name=tool_name,
                signal="failing",
                confidence=0.7,
                org_id="canary_prober",
                weight=self.penalty_weight,
            )
            self.hub.contribute(penalty_packet, anonymize=False)
            penalty_applied = True

        result = CanaryResult(
            tool_name=tool_name,
            entropy_score_before=entropy,
            adversarial_suspect=adversarial_suspect,
            timestamp=time.time(),
            probe_action=probe_action,
            penalty_applied=penalty_applied,
        )
        with self._lock:
            self._results.append(result)
        return result

    def probe_high_entropy_tools(self) -> list[CanaryResult]:
        """Probe all tools with entropy ≥ ``entropy_threshold``.

        Returns
        -------
        list[CanaryResult]
            One result per probed tool (empty if none exceed the threshold).
        """
        all_entropy = self.decay.all_tool_entropy()
        targets = [
            name
            for name, score in all_entropy.items()
            if score >= self.entropy_threshold
        ]
        return [self.probe_once(name) for name in targets]

    # ------------------------------------------------------------------
    # Background daemon
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background canary daemon (non-blocking).

        Calling ``start()`` when already running is a no-op.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="manifold-canary-prober",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background daemon to stop.

        The currently-sleeping interval will complete before the thread exits.
        """
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        """Return ``True`` if the background daemon is active."""
        return self._running

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def probe_results(self) -> list[CanaryResult]:
        """Return a copy of all recorded canary probe results."""
        with self._lock:
            return list(self._results)

    def canary_summary(self) -> dict[str, Any]:
        """Return a lightweight summary of canary activity.

        Keys: ``total_probes``, ``adversarial_suspects``, ``penalties_applied``,
        ``is_running``, ``pass_rate``.
        """
        with self._lock:
            total = len(self._results)
            suspects = sum(1 for r in self._results if r.adversarial_suspect)
            penalties = sum(1 for r in self._results if r.penalty_applied)
            passes = sum(1 for r in self._results if r.probe_action == "pass")
        return {
            "total_probes": total,
            "adversarial_suspects": suspects,
            "penalties_applied": penalties,
            "is_running": self._running,
            "pass_rate": passes / total if total > 0 else 1.0,
        }

    def latest_result(self, tool_name: str) -> CanaryResult | None:
        """Return the most recent ``CanaryResult`` for *tool_name*, or ``None``."""
        with self._lock:
            for result in reversed(self._results):
                if result.tool_name == tool_name:
                    return result
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background probe loop — runs until ``stop()`` is called."""
        while True:
            with self._lock:
                if not self._running:
                    break
            try:
                self.probe_high_entropy_tools()
            except Exception:  # noqa: BLE001
                pass  # never crash the daemon thread
            time.sleep(self.interval_seconds)
