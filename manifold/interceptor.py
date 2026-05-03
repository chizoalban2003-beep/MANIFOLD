"""Phase 13: Active Interceptor — MANIFOLD's pre-flight safety veto.

Once MANIFOLD has accumulated enough shadow data, the ``ActiveInterceptor``
acts as a **Gatekeeper** between the caller and the physical tool execution.
It performs a pre-flight risk check and either permits the call, redirects it
to the next-best tool, or escalates to the HITL gate — before a single byte
leaves the system.

Key decision rule
-----------------
.. math::

    Decision_{Final} =
    \\begin{cases}
        \\text{Execute}  & \\text{if } (Risk \\times Stakes) < Threshold \\\\
        \\text{Redirect} & \\text{otherwise}
    \\end{cases}

Three outcomes on veto
----------------------
1. **Redirect to HITL** (default) — the task is flagged for human review.
2. **Fallback to next-best tool** — the registry is searched for the tool
   with the highest ``reliability – risk`` score that is *not* the vetoed one.
3. **Hard refusal** — raises ``InterceptorVeto`` (callers must catch it).

``@shield`` decorator
---------------------
A lightweight decorator that wraps any Python callable with a MANIFOLD
pre-flight check.  If the brain decides the risk is too high, the wrapped
function is never called.::

    @shield(brain=brain, domain="finance", stakes=0.8)
    def process_payment(amount: float) -> dict:
        # Only executes if MANIFOLD approves the risk profile
        return payment_api.charge(amount)

Key classes
-----------
``InterceptorConfig``
    Tunable thresholds and redirect strategy.
``InterceptResult``
    The outcome of a pre-flight check (permit / redirect / refuse).
``InterceptorVeto``
    Exception raised when the interceptor blocks a call in ``raise_on_veto``
    mode.
``ActiveInterceptor``
    The main gatekeeper: wraps ``ConnectorRegistry`` calls with the veto gate.
``shield``
    Function decorator for zero-boilerplate MANIFOLD integration.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

from .brain import BrainDecision, BrainTask, ManifoldBrain
from .connector import ConnectorRegistry, ToolConnectorResult
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# InterceptorConfig
# ---------------------------------------------------------------------------


@dataclass
class InterceptorConfig:
    """Configuration for the ``ActiveInterceptor``.

    Parameters
    ----------
    risk_veto_threshold:
        Block a tool call when ``task.stakes * tool.risk >= threshold``.
        Default: ``0.45`` (moderate; lower = stricter).
    redirect_strategy:
        What to do when a call is vetoed:

        * ``"hitl"``     — escalate to human-in-the-loop (default).
        * ``"fallback"`` — silently route to the next-best tool in the registry.
        * ``"refuse"``   — raise ``InterceptorVeto``.
    fallback_min_reliability:
        Minimum reliability a fallback tool must have to be eligible.
        Default: ``0.70``.
    """

    risk_veto_threshold: float = 0.45
    redirect_strategy: str = "hitl"  # "hitl" | "fallback" | "refuse"
    fallback_min_reliability: float = 0.70


# ---------------------------------------------------------------------------
# InterceptResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterceptResult:
    """Outcome of a single pre-flight intercept check.

    Attributes
    ----------
    tool_name:
        The originally requested tool.
    permitted:
        ``True`` if the call may proceed as-is.
    risk_score:
        ``task.stakes * tool.risk`` at intercept time.
    redirect_to:
        Name of the redirect target (``"hitl"`` for HITL escalation, or the
        fallback tool name), or ``None`` if the call was permitted.
    reason:
        Human-readable explanation of the decision.
    manifold_decision:
        The underlying ``BrainDecision`` (available even on permits).
    """

    tool_name: str
    permitted: bool
    risk_score: float
    redirect_to: str | None
    reason: str
    manifold_decision: BrainDecision


# ---------------------------------------------------------------------------
# InterceptorVeto
# ---------------------------------------------------------------------------


class InterceptorVeto(RuntimeError):
    """Raised when the ``ActiveInterceptor`` blocks a call in *refuse* mode.

    Attributes
    ----------
    result:
        The ``InterceptResult`` that triggered the veto.
    """

    def __init__(self, message: str, result: InterceptResult) -> None:
        super().__init__(message)
        self.result = result


# ---------------------------------------------------------------------------
# ActiveInterceptor
# ---------------------------------------------------------------------------


@dataclass
class ActiveInterceptor:
    """Pre-flight gatekeeper that wraps ``ConnectorRegistry`` tool calls.

    ``ActiveInterceptor`` sits between the caller and the physical tool
    execution.  For every requested tool call it:

    1. Retrieves the tool's current ``ToolProfile`` from the registry.
    2. Asks ``ManifoldBrain`` to decide the task.
    3. Computes ``risk_score = task.stakes * tool.risk``.
    4. If ``risk_score >= config.risk_veto_threshold`` *or* the brain says
       ``refuse`` / ``escalate``, applies the configured redirect strategy.

    Parameters
    ----------
    registry:
        The ``ConnectorRegistry`` containing all available tools.
    brain:
        A ``ManifoldBrain`` used for decision-making.
    config:
        Tunable intercept parameters (see ``InterceptorConfig``).

    Example
    -------
    ::

        interceptor = ActiveInterceptor(registry=registry, brain=brain)
        result = interceptor.intercept(task, requested_tool="billing_api")
        if result.permitted:
            # proceed with connector.call(...)
            connector = registry.get(result.tool_name)
            ...
        else:
            print(f"Vetoed — redirect to {result.redirect_to}")
    """

    registry: ConnectorRegistry
    brain: ManifoldBrain
    config: InterceptorConfig = field(default_factory=InterceptorConfig)

    _intercept_log: list[InterceptResult] = field(
        default_factory=list, init=False, repr=False
    )

    def intercept(self, task: BrainTask, requested_tool: str) -> InterceptResult:
        """Perform a pre-flight check for a tool call.

        Parameters
        ----------
        task:
            The task being evaluated.
        requested_tool:
            Name of the tool the caller wants to use.

        Returns
        -------
        InterceptResult
            Describes whether the call is permitted and any redirect target.

        Raises
        ------
        KeyError
            If *requested_tool* is not registered and no fallback is possible.
        InterceptorVeto
            If the strategy is ``"refuse"`` and the risk threshold is breached.
        """
        connector = self.registry.get(requested_tool)
        if connector is None:
            raise KeyError(f"Tool {requested_tool!r} is not registered.")

        profile = connector.refreshed_profile()
        decision = self.brain.decide(task)
        risk_score = clamp01(task.stakes * profile.risk)

        # Permit if risk is below threshold and brain does not veto
        brain_veto = decision.action in {"refuse", "escalate"}
        threshold_breach = risk_score >= self.config.risk_veto_threshold

        if not brain_veto and not threshold_breach:
            result = InterceptResult(
                tool_name=requested_tool,
                permitted=True,
                risk_score=risk_score,
                redirect_to=None,
                reason="risk within threshold",
                manifold_decision=decision,
            )
            self._intercept_log.append(result)
            return result

        # Build veto reason
        if brain_veto:
            reason = f"brain action={decision.action!r}"
        else:
            reason = (
                f"risk_score={risk_score:.3f} >= threshold="
                f"{self.config.risk_veto_threshold:.3f}"
            )

        redirect_to = self._resolve_redirect(requested_tool)

        if self.config.redirect_strategy == "refuse" and redirect_to is None:
            result = InterceptResult(
                tool_name=requested_tool,
                permitted=False,
                risk_score=risk_score,
                redirect_to=None,
                reason=reason,
                manifold_decision=decision,
            )
            self._intercept_log.append(result)
            raise InterceptorVeto(
                f"MANIFOLD vetoed {requested_tool!r}: {reason}", result
            )

        result = InterceptResult(
            tool_name=requested_tool,
            permitted=False,
            risk_score=risk_score,
            redirect_to=redirect_to,
            reason=reason,
            manifold_decision=decision,
        )
        self._intercept_log.append(result)
        return result

    def call(
        self,
        task: BrainTask,
        requested_tool: str,
        *args: Any,
        **kwargs: Any,
    ) -> ToolConnectorResult:
        """Intercept and (if permitted) execute a tool call.

        This is the single entry-point for callers who want MANIFOLD to govern
        every tool call without manual ``intercept()`` / ``get()`` steps.

        * If the call is **permitted**: executes via the requested connector.
        * If **redirected to HITL**: returns a synthetic failure result with
          ``error_type="hitl_escalation"`` so the caller can detect it.
        * If **redirected to fallback tool**: executes via the fallback connector.
        * If **refused** (strategy=``"refuse"``): raises ``InterceptorVeto``.

        Parameters
        ----------
        task:
            The task governing the risk assessment.
        requested_tool:
            The tool the caller originally wanted to use.
        *args, **kwargs:
            Passed to the connector's ``call()`` method.

        Returns
        -------
        ToolConnectorResult
        """
        result = self.intercept(task, requested_tool)

        if result.permitted:
            connector = self.registry.get(requested_tool)
            assert connector is not None  # guaranteed by intercept()
            return connector.call(*args, **kwargs)

        # Redirect
        target = result.redirect_to
        if target == "hitl" or target is None:
            return ToolConnectorResult(
                tool_name=requested_tool,
                output=None,
                success=False,
                latency_seconds=0.0,
                error_type="hitl_escalation",
                error_message=f"Interceptor redirected to HITL: {result.reason}",
            )

        # Fallback to another registered tool
        fallback_connector = self.registry.get(target)
        if fallback_connector is None:
            return ToolConnectorResult(
                tool_name=requested_tool,
                output=None,
                success=False,
                latency_seconds=0.0,
                error_type="hitl_escalation",
                error_message=f"Fallback {target!r} not found; escalated to HITL",
            )
        return fallback_connector.call(*args, **kwargs)

    def _resolve_redirect(self, vetoed_tool: str) -> str | None:
        """Return the redirect target based on the configured strategy.

        Parameters
        ----------
        vetoed_tool:
            Name of the tool that was blocked.

        Returns
        -------
        str | None
            ``"hitl"`` for HITL escalation, a fallback tool name, or ``None``
            when the strategy is ``"refuse"``.
        """
        if self.config.redirect_strategy == "hitl":
            return "hitl"
        if self.config.redirect_strategy == "refuse":
            return None
        # "fallback" — find the best alternative tool in the registry
        best_name: str | None = None
        best_score: float = -1.0
        for name in self.registry.names():
            if name == vetoed_tool:
                continue
            connector = self.registry.get(name)
            if connector is None:
                continue
            profile = connector.refreshed_profile()
            if profile.reliability < self.config.fallback_min_reliability:
                continue
            score = profile.reliability - profile.risk
            if score > best_score:
                best_score = score
                best_name = name
        return best_name if best_name is not None else "hitl"

    def intercept_log(self) -> list[InterceptResult]:
        """Return a copy of all intercept decisions made so far."""
        return list(self._intercept_log)

    def veto_count(self) -> int:
        """Return the number of vetoed (not permitted) calls."""
        return sum(1 for r in self._intercept_log if not r.permitted)

    def permit_count(self) -> int:
        """Return the number of permitted calls."""
        return sum(1 for r in self._intercept_log if r.permitted)

    def veto_rate(self) -> float:
        """Return the fraction of calls that were vetoed."""
        total = len(self._intercept_log)
        if total == 0:
            return 0.0
        return self.veto_count() / total

    def summary(self) -> dict[str, object]:
        """Return a summary of interceptor activity.

        Returns
        -------
        dict
            Keys: ``total_calls``, ``permitted``, ``vetoed``, ``veto_rate``,
            ``redirected_to_hitl``, ``redirected_to_fallback``,
            ``avg_risk_score``.
        """
        log = self._intercept_log
        hitl_count = sum(1 for r in log if not r.permitted and r.redirect_to == "hitl")
        fallback_count = sum(
            1 for r in log
            if not r.permitted and r.redirect_to not in (None, "hitl")
        )
        avg_risk = (
            sum(r.risk_score for r in log) / len(log) if log else 0.0
        )
        return {
            "total_calls": len(log),
            "permitted": self.permit_count(),
            "vetoed": self.veto_count(),
            "veto_rate": self.veto_rate(),
            "redirected_to_hitl": hitl_count,
            "redirected_to_fallback": fallback_count,
            "avg_risk_score": avg_risk,
        }


# ---------------------------------------------------------------------------
# @shield decorator
# ---------------------------------------------------------------------------


def shield(
    brain: ManifoldBrain,
    *,
    domain: str = "general",
    stakes: float = 0.5,
    uncertainty: float = 0.5,
    complexity: float = 0.5,
    safety_sensitivity: float = 0.3,
    veto_actions: tuple[str, ...] = ("refuse", "escalate"),
    raise_on_veto: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps a function with a MANIFOLD pre-flight check.

    If the brain decides the risk is too high (action in *veto_actions*),
    the wrapped function is never called.

    Parameters
    ----------
    brain:
        A ``ManifoldBrain`` instance used for risk assessment.
    domain:
        Task domain for the pre-flight check (e.g. ``"finance"``).
    stakes:
        Task stakes (0–1).  High stakes → stricter veto.
    uncertainty:
        Task uncertainty (0–1).
    complexity:
        Task complexity (0–1).
    safety_sensitivity:
        Safety sensitivity (0–1).
    veto_actions:
        Brain actions that trigger a veto.  Default: ``("refuse", "escalate")``.
    raise_on_veto:
        If ``True`` (default), raises ``InterceptorVeto`` on veto.
        If ``False``, returns ``None`` silently.

    Returns
    -------
    Callable
        A decorated callable that is guarded by the MANIFOLD pre-flight check.

    Example
    -------
    ::

        @shield(brain=brain, domain="finance", stakes=0.8)
        def process_payment(amount: float) -> dict:
            return payment_api.charge(amount)

        try:
            result = process_payment(500.0)
        except InterceptorVeto as exc:
            print(f"MANIFOLD blocked the payment: {exc}")
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            task = BrainTask(
                prompt=fn.__name__,
                domain=domain,
                stakes=stakes,
                uncertainty=uncertainty,
                complexity=complexity,
                safety_sensitivity=safety_sensitivity,
            )
            decision = brain.decide(task)
            if decision.action in veto_actions:
                fake_result = InterceptResult(
                    tool_name=fn.__name__,
                    permitted=False,
                    risk_score=clamp01(stakes * safety_sensitivity),
                    redirect_to="hitl",
                    reason=f"brain action={decision.action!r}",
                    manifold_decision=decision,
                )
                if raise_on_veto:
                    raise InterceptorVeto(
                        f"MANIFOLD @shield blocked {fn.__name__!r}: "
                        f"action={decision.action!r}",
                        fake_result,
                    )
                return None
            return fn(*args, **kwargs)

        return wrapper

    return decorator
