"""Phase 39: Autonomic PID Controller — Dynamic Risk Self-Regulation.

``RiskPIDController`` implements a standard proportional-integral-derivative
(PID) control loop that adjusts the
:class:`~manifold.interceptor.InterceptorConfig` risk threshold in real-time
based on the global system entropy from the
:class:`~manifold.hub.ReputationHub`.

The PID formula
---------------
.. math::

    u(t) = K_p e(t) + K_i \\int e(t)\\,dt + K_d \\frac{de(t)}{dt}

where :math:`e(t) = \\text{setpoint} - \\text{entropy}` is the error signal.

Anti-windup
-----------
The integral term is clamped to ``[-integral_limit, +integral_limit]``
preventing *integral windup* when the system is saturated.

Key classes
-----------
``PIDConfig``
    Tunable PID gains and limits.
``PIDState``
    Immutable snapshot of controller state at one tick.
``RiskPIDController``
    The main control-loop.  Call :meth:`tick` periodically or on each request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# PIDConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIDConfig:
    """Configuration for :class:`RiskPIDController`.

    Parameters
    ----------
    kp:
        Proportional gain.  Default: ``1.0``.
    ki:
        Integral gain.  Default: ``0.1``.
    kd:
        Derivative gain.  Default: ``0.05``.
    setpoint:
        Target system entropy (the "desired" value).  When entropy equals
        *setpoint*, the controller output is zero.  Default: ``0.3``.
    output_min:
        Minimum allowable output (interceptor threshold lower bound).
        Default: ``0.1``.
    output_max:
        Maximum allowable output (interceptor threshold upper bound).
        Default: ``0.9``.
    integral_limit:
        Anti-windup clamp applied to the integral term.  Default: ``5.0``.
    """

    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    setpoint: float = 0.3
    output_min: float = 0.1
    output_max: float = 0.9
    integral_limit: float = 5.0


# ---------------------------------------------------------------------------
# PIDState — immutable snapshot per tick
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIDState:
    """Immutable snapshot of controller state after one :meth:`~RiskPIDController.tick`.

    Attributes
    ----------
    timestamp:
        POSIX timestamp when this tick was computed.
    measured_value:
        The system entropy value measured at this tick.
    error:
        ``setpoint - measured_value``.
    proportional:
        Proportional term ``K_p * error``.
    integral:
        Accumulated integral term (after anti-windup clamp).
    derivative:
        Derivative term ``K_d * (error - prev_error) / dt``.
    output:
        Final controller output (clamped to ``[output_min, output_max]``).
    threshold_before:
        Interceptor threshold before applying the controller output.
    threshold_after:
        New interceptor threshold after applying the controller output.
    """

    timestamp: float
    measured_value: float
    error: float
    proportional: float
    integral: float
    derivative: float
    output: float
    threshold_before: float
    threshold_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "measured_value": round(self.measured_value, 6),
            "error": round(self.error, 6),
            "proportional": round(self.proportional, 6),
            "integral": round(self.integral, 6),
            "derivative": round(self.derivative, 6),
            "output": round(self.output, 6),
            "threshold_before": round(self.threshold_before, 6),
            "threshold_after": round(self.threshold_after, 6),
        }


# ---------------------------------------------------------------------------
# RiskPIDController
# ---------------------------------------------------------------------------


@dataclass
class RiskPIDController:
    """Autonomic PID controller for MANIFOLD's risk threshold.

    Reads the current system entropy from a
    :class:`~manifold.hub.ReputationHub` (or any callable that returns a
    float in ``[0, 1]``) and adjusts the
    :class:`~manifold.interceptor.InterceptorConfig` ``risk_veto_threshold``
    accordingly.

    When system entropy rises above the *setpoint* (i.e. AI models are
    degrading), the controller *lowers* the threshold — making MANIFOLD
    stricter.  When entropy falls below the setpoint the threshold rises,
    relaxing controls.

    Parameters
    ----------
    config:
        PID tuning parameters.
    interceptor_config:
        The ``InterceptorConfig`` whose ``risk_veto_threshold`` attribute
        will be dynamically updated.  When ``None``, the controller runs in
        *read-only* mode (useful for testing).
    entropy_source:
        A zero-argument callable returning the current system entropy.
        Defaults to ``lambda: 0.0`` (static source for headless use).

    Example
    -------
    ::

        pid = RiskPIDController(
            config=PIDConfig(kp=1.5, ki=0.1, kd=0.02, setpoint=0.25),
            interceptor_config=interceptor.config,
            entropy_source=hub.system_entropy,
        )
        state = pid.tick()
        print(f"New threshold: {state.threshold_after:.4f}")
    """

    config: PIDConfig = field(default_factory=PIDConfig)
    interceptor_config: Any = None  # InterceptorConfig | None
    entropy_source: Any = field(default=None)  # Callable[[], float] | None

    _integral: float = field(default=0.0, init=False)
    _prev_error: float | None = field(default=None, init=False)
    _prev_time: float | None = field(default=None, init=False)
    _history: list[PIDState] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.entropy_source is None:
            self.entropy_source = lambda: 0.0

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def tick(self, measured_value: float | None = None) -> PIDState:
        """Advance the controller by one step.

        Parameters
        ----------
        measured_value:
            Optional override for the measured entropy.  When ``None``
            (default), :attr:`entropy_source` is called to obtain the
            current value.

        Returns
        -------
        PIDState
            Snapshot of this tick, including the new threshold.
        """
        now = time.monotonic()
        if measured_value is None:
            measured_value = float(self.entropy_source())

        # Compute dt (seconds since last tick; 1.0 for first tick)
        dt = 1.0
        if self._prev_time is not None:
            dt = max(1e-6, now - self._prev_time)

        error = self.config.setpoint - measured_value

        # --- Proportional ---
        p_term = self.config.kp * error

        # --- Integral with anti-windup ---
        self._integral += error * dt
        # Clamp to prevent integral windup
        limit = self.config.integral_limit
        self._integral = max(-limit, min(limit, self._integral))
        i_term = self.config.ki * self._integral

        # --- Derivative ---
        if self._prev_error is None:
            d_term = 0.0
        else:
            d_term = self.config.kd * (error - self._prev_error) / dt

        raw_output = p_term + i_term + d_term
        output = max(
            self.config.output_min,
            min(self.config.output_max, raw_output),
        )

        # --- Read current threshold (before) ---
        threshold_before = (
            self.interceptor_config.risk_veto_threshold
            if self.interceptor_config is not None
            else 0.45
        )

        # --- Apply new threshold ---
        if self.interceptor_config is not None:
            self.interceptor_config.risk_veto_threshold = output
        threshold_after = output

        # --- Update controller state ---
        self._prev_error = error
        self._prev_time = now

        state = PIDState(
            timestamp=time.time(),
            measured_value=measured_value,
            error=error,
            proportional=p_term,
            integral=i_term,
            derivative=d_term,
            output=output,
            threshold_before=threshold_before,
            threshold_after=threshold_after,
        )
        self._history.append(state)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset integrator and history; useful for test scenarios."""
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None
        self._history.clear()

    def history(self) -> list[PIDState]:
        """Return all recorded PID states (most recent last)."""
        return list(self._history)

    def last_state(self) -> PIDState | None:
        """Return the most recent :class:`PIDState`, or ``None``."""
        return self._history[-1] if self._history else None

    def summary(self) -> dict[str, Any]:
        """Return a summary of controller activity.

        Returns
        -------
        dict
            Keys: ``total_ticks``, ``integral``, ``last_output``,
            ``last_error``, ``setpoint``, ``kp``, ``ki``, ``kd``.
        """
        last = self.last_state()
        return {
            "total_ticks": len(self._history),
            "integral": round(self._integral, 6),
            "last_output": round(last.output, 6) if last else 0.0,
            "last_error": round(last.error, 6) if last else 0.0,
            "setpoint": self.config.setpoint,
            "kp": self.config.kp,
            "ki": self.config.ki,
            "kd": self.config.kd,
        }
