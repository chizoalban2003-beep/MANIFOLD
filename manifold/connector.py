"""Phase 8: Standard Connector — universal tool adapter for MANIFOLD.

``ToolConnector`` wraps any Python callable (REST handler, LangChain tool,
function, or class method) as a MANIFOLD ``ToolProfile`` + execution engine.
It automatically captures telemetry (latency, success/failure, exceptions)
and converts it into ``BrainOutcome`` objects that feed back into
``PriceAdapter`` and ``BrainMemory``.

The design principle: **zero code changes** in existing agent stacks.  Drop
a ``ToolConnector`` around any tool you already have and MANIFOLD starts
learning from it immediately.

``ShadowModeWrapper`` provides a non-interventional observation layer that
lets MANIFOLD shadow an existing agent workflow — logging virtual regret
and computing counterfactual decisions — without changing any live behaviour.
Once shadow data accumulates, the operator can flip a flag to let MANIFOLD
take control.

Key classes
-----------
``ToolConnectorResult``
    Result of executing a connected tool, including timing and telemetry.
``ToolConnector``
    Wraps any callable as a MANIFOLD tool.  Handles success/failure capture,
    latency measurement, and ``BrainOutcome`` generation.
``ShadowModeWrapper``
    Observes existing agent decisions, records ``VirtualRegret``, and logs
    what MANIFOLD *would* have decided without intervening.
``VirtualRegret``
    A record of a shadow observation: actual vs. MANIFOLD-counterfactual decision.
``ConnectorRegistry``
    Manages a named registry of ``ToolConnector`` instances.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .brain import (
    BrainConfig,
    BrainDecision,
    BrainOutcome,
    BrainTask,
    ManifoldBrain,
    ToolProfile,
    default_tools,
)
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# ToolConnectorResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolConnectorResult:
    """Result of executing a ``ToolConnector``-wrapped callable.

    Attributes
    ----------
    tool_name:
        Name of the wrapped tool.
    output:
        Raw return value from the underlying callable, or ``None`` on failure.
    success:
        Whether the call completed without raising an exception.
    latency_seconds:
        Wall-clock time taken by the call, in seconds.
    error_type:
        Exception class name if the call failed, or ``""`` on success.
    error_message:
        Short error message string, or ``""`` on success.
    """

    tool_name: str
    output: Any
    success: bool
    latency_seconds: float
    error_type: str = ""
    error_message: str = ""

    def to_brain_outcome(
        self,
        *,
        cost_per_second: float = 0.05,
        asset_on_success: float = 0.75,
        base_risk: float = 0.05,
    ) -> BrainOutcome:
        """Convert telemetry to a ``BrainOutcome`` for MANIFOLD learning.

        Parameters
        ----------
        cost_per_second:
            How much latency costs per second (default: 0.05).
        asset_on_success:
            Asset gained on a successful call (default: 0.75).
        base_risk:
            Baseline risk realized on failure (default: 0.05).

        Returns
        -------
        BrainOutcome
        """
        cost_paid = clamp01(self.latency_seconds * cost_per_second)
        if self.success:
            return BrainOutcome(
                success=True,
                cost_paid=cost_paid,
                risk_realized=0.0,
                asset_gained=asset_on_success,
            )
        failure_mode = _classify_error(self.error_type)
        return BrainOutcome(
            success=False,
            cost_paid=cost_paid,
            risk_realized=clamp01(base_risk + 0.30),
            asset_gained=0.0,
            failure_mode=failure_mode,
        )


def _classify_error(error_type: str) -> str:
    """Map a Python exception class name to a MANIFOLD failure mode."""
    mapping = {
        "TimeoutError": "timeout",
        "ConnectionError": "tool_error",
        "ConnectionRefusedError": "tool_error",
        "HTTPError": "tool_error",
        "ValueError": "bad_data",
        "KeyError": "bad_data",
        "AttributeError": "bad_data",
        "TypeError": "bad_data",
        "NotImplementedError": "tool_error",
        "RateLimitError": "rate_limit",
        "PermissionError": "tool_error",
    }
    return mapping.get(error_type, "unknown")


# ---------------------------------------------------------------------------
# ToolConnector
# ---------------------------------------------------------------------------


@dataclass
class ToolConnector:
    """Wraps any Python callable as a MANIFOLD tool with telemetry capture.

    ``ToolConnector`` is the standard integration point for existing tools,
    LangChain tools, REST handlers, and any other callables.  It:

    1. Executes the callable and captures latency and success/failure.
    2. Converts the result to a ``BrainOutcome`` for MANIFOLD learning.
    3. Exposes a ``ToolProfile`` so the tool can be passed to ``ManifoldBrain``.

    Parameters
    ----------
    name:
        Human-readable tool name (must be unique in a ``ConnectorRegistry``).
    fn:
        The callable to wrap.
    profile:
        Optional ``ToolProfile`` override.  If not provided, a default profile
        is created from the name with neutral parameters.
    cost_per_second:
        Telemetry cost multiplier for latency.
    asset_on_success:
        Asset gained when the call succeeds.
    base_risk:
        Baseline risk on failure.

    Example
    -------
    ::

        connector = ToolConnector(
            name="my_api",
            fn=lambda query: requests.get(url, params={"q": query}).json(),
            profile=ToolProfile("my_api", cost=0.10, latency=0.15, reliability=0.85, risk=0.10, asset=0.70),
        )
        result = connector.call("search for legal precedents")
        outcome = result.to_brain_outcome()
    """

    name: str
    fn: Callable[..., Any]
    profile: ToolProfile = field(default=None)  # type: ignore[assignment]
    cost_per_second: float = 0.05
    asset_on_success: float = 0.75
    base_risk: float = 0.05

    _call_count: int = field(default=0, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _total_latency: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.profile is None:
            self.profile = ToolProfile(
                self.name,
                cost=0.10,
                latency=0.15,
                reliability=0.80,
                risk=0.10,
                asset=0.70,
            )

    def call(self, *args: Any, **kwargs: Any) -> ToolConnectorResult:
        """Execute the wrapped callable and capture telemetry.

        Parameters
        ----------
        *args, **kwargs:
            Passed directly to the underlying callable.

        Returns
        -------
        ToolConnectorResult
            Contains output, timing, and success/failure information.
        """
        start = time.monotonic()
        try:
            output = self.fn(*args, **kwargs)
            latency = time.monotonic() - start
            self._call_count += 1
            self._total_latency += latency
            return ToolConnectorResult(
                tool_name=self.name,
                output=output,
                success=True,
                latency_seconds=latency,
            )
        except Exception as exc:
            latency = time.monotonic() - start
            self._call_count += 1
            self._failure_count += 1
            self._total_latency += latency
            return ToolConnectorResult(
                tool_name=self.name,
                output=None,
                success=False,
                latency_seconds=latency,
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
            )

    def observed_reliability(self) -> float:
        """Return the observed success rate based on captured telemetry.

        Returns 1.0 (perfect) if no calls have been made yet.
        """
        if self._call_count == 0:
            return 1.0
        return clamp01(1.0 - self._failure_count / self._call_count)

    def mean_latency(self) -> float:
        """Return the mean observed latency in seconds."""
        if self._call_count == 0:
            return 0.0
        return self._total_latency / self._call_count

    def refreshed_profile(self) -> ToolProfile:
        """Return a new ``ToolProfile`` with telemetry-updated reliability and latency.

        Blends stated profile values with observed telemetry using a 50/50 weight.
        After enough calls, the observed values dominate.
        """
        import dataclasses as _dc
        obs_rel = self.observed_reliability()
        obs_lat = min(1.0, self.mean_latency())
        blend = min(1.0, self._call_count / 20.0)  # reaches full weight at 20 calls
        new_reliability = clamp01(
            (1.0 - blend) * self.profile.reliability + blend * obs_rel
        )
        new_latency = (1.0 - blend) * self.profile.latency + blend * obs_lat
        return _dc.replace(self.profile, reliability=new_reliability, latency=new_latency)

    def call_count(self) -> int:
        """Return total number of calls (successes + failures)."""
        return self._call_count

    def failure_count(self) -> int:
        """Return total number of failed calls."""
        return self._failure_count


# ---------------------------------------------------------------------------
# ConnectorRegistry
# ---------------------------------------------------------------------------


@dataclass
class ConnectorRegistry:
    """Named registry of ``ToolConnector`` instances.

    Provides a central location for registering, looking up, and refreshing
    all tool connectors in an agent ecosystem.

    Example
    -------
    ::

        registry = ConnectorRegistry()
        registry.register(ToolConnector("search", fn=my_search_fn))
        registry.register(ToolConnector("calc", fn=my_calc_fn))

        # Get current ToolProfile list for ManifoldBrain
        tools = registry.tool_profiles()
        brain = ManifoldBrain(config, tools=tools)
    """

    _connectors: dict[str, ToolConnector] = field(default_factory=dict, init=False, repr=False)

    def register(self, connector: ToolConnector) -> None:
        """Register a connector under its name.  Overwrites any existing entry."""
        self._connectors[connector.name] = connector

    def get(self, name: str) -> ToolConnector | None:
        """Return the connector for *name*, or ``None`` if not registered."""
        return self._connectors.get(name)

    def names(self) -> list[str]:
        """Return all registered connector names."""
        return list(self._connectors.keys())

    def tool_profiles(self, *, use_telemetry: bool = True) -> list[ToolProfile]:
        """Return a list of ``ToolProfile`` instances for all registered connectors.

        Parameters
        ----------
        use_telemetry:
            If ``True``, returns ``refreshed_profile()`` (telemetry-blended).
            If ``False``, returns the stated profile as-is.
        """
        if use_telemetry:
            return [c.refreshed_profile() for c in self._connectors.values()]
        return [c.profile for c in self._connectors.values()]

    def __len__(self) -> int:
        return len(self._connectors)


# ---------------------------------------------------------------------------
# VirtualRegret + ShadowModeWrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VirtualRegret:
    """A single shadow observation: actual vs. MANIFOLD counterfactual.

    Attributes
    ----------
    task:
        The ``BrainTask`` being observed.
    actual_action:
        The action the existing agent actually took (passed in by caller).
    manifold_action:
        The action MANIFOLD *would* have chosen.
    manifold_decision:
        Full ``BrainDecision`` from MANIFOLD.
    regret:
        ``1`` if actual != manifold (MANIFOLD would have decided differently),
        ``0`` if they agree.
    """

    task: BrainTask
    actual_action: str
    manifold_action: str
    manifold_decision: BrainDecision
    regret: int

    @property
    def is_disagreement(self) -> bool:
        """True when MANIFOLD would have chosen a different action."""
        return self.regret == 1


@dataclass
class ShadowModeWrapper:
    """Non-interventional MANIFOLD observer for existing agent workflows.

    ``ShadowModeWrapper`` shadows an existing agent decision process.  For
    every task that passes through it, MANIFOLD computes a counterfactual
    decision (what it *would* have done) without changing the actual execution.

    Once enough shadow data accumulates, the operator can:
    1. Review the ``virtual_regret_log`` to assess potential improvement.
    2. Call ``activate()`` to flip MANIFOLD from passive to active.

    Parameters
    ----------
    brain:
        A ``ManifoldBrain`` instance MANIFOLD uses for counterfactual decisions.
    active:
        Whether MANIFOLD is actively controlling decisions (default: False = shadow).

    Example
    -------
    ::

        wrapper = ShadowModeWrapper(brain=ManifoldBrain(cfg, tools=tools))

        # In your existing agent loop:
        actual_action = existing_agent.decide(task)
        wrapper.observe(task, actual_action=actual_action)

        # After shadow period:
        report = wrapper.shadow_report()
        print(f"MANIFOLD would disagree {report['disagreement_rate']:.1%} of the time")

        # Flip to active:
        wrapper.activate()
    """

    brain: ManifoldBrain
    active: bool = False

    _regret_log: list[VirtualRegret] = field(default_factory=list, init=False, repr=False)

    def observe(self, task: BrainTask, actual_action: str) -> VirtualRegret:
        """Shadow one agent decision.

        Computes MANIFOLD's counterfactual decision without executing it, then
        records a ``VirtualRegret`` entry for post-hoc analysis.

        Parameters
        ----------
        task:
            The task the existing agent is handling.
        actual_action:
            The action the existing agent actually chose (string).

        Returns
        -------
        VirtualRegret
            The shadow record for this observation.
        """
        manifold_decision = self.brain.decide(task)
        regret = 0 if manifold_decision.action == actual_action else 1
        vr = VirtualRegret(
            task=task,
            actual_action=actual_action,
            manifold_action=manifold_decision.action,
            manifold_decision=manifold_decision,
            regret=regret,
        )
        self._regret_log.append(vr)
        return vr

    def activate(self) -> None:
        """Switch from shadow mode to active mode.

        In active mode, callers should use ``brain.decide()`` directly
        instead of relying on the existing agent.
        """
        self.active = True

    def deactivate(self) -> None:
        """Revert to shadow/observation mode."""
        self.active = False

    def virtual_regret_log(self) -> list[VirtualRegret]:
        """Return a copy of all shadow observations."""
        return list(self._regret_log)

    def total_regret(self) -> int:
        """Total number of disagreements observed so far."""
        return sum(r.regret for r in self._regret_log)

    def disagreement_rate(self) -> float:
        """Fraction of tasks where MANIFOLD would have decided differently."""
        if not self._regret_log:
            return 0.0
        return self.total_regret() / len(self._regret_log)

    def shadow_report(self) -> dict[str, object]:
        """Return a summary dict of the shadow observation window.

        Keys:
        - ``total_observations``: int
        - ``total_disagreements``: int
        - ``disagreement_rate``: float
        - ``top_disagreement_actions``: list of (actual, manifold) pairs (most common)
        - ``active``: bool
        """
        actions: dict[tuple[str, str], int] = {}
        for vr in self._regret_log:
            if vr.is_disagreement:
                key = (vr.actual_action, vr.manifold_action)
                actions[key] = actions.get(key, 0) + 1
        top = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "total_observations": len(self._regret_log),
            "total_disagreements": self.total_regret(),
            "disagreement_rate": self.disagreement_rate(),
            "top_disagreement_actions": [(a, m) for (a, m), _ in top],
            "active": self.active,
        }
