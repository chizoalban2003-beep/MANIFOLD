"""Phase 26: Reputation Decay & Entropy — Model Drift Awareness.

Tracks the *age* of the last gossip/outcome signal for each tool and applies
a time-variant exponential decay to its reliability score:

.. math::

    Rep_t = Rep_{t-1} \\cdot e^{-\\lambda \\Delta t}

The decay rate :math:`\\lambda` (volatility coefficient) is domain-specific:
high-churn domains like ``llm`` decay faster than stable domains like ``math``
or ``storage``.

Key classes
-----------
``VolatilityTable``
    Maps tool domains to their :math:`\\lambda` decay coefficients.
``ReputationDecay``
    Stateful decay engine that timestamps signals and applies decay on query.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Final


# ---------------------------------------------------------------------------
# Default per-domain volatility coefficients (λ)
# ---------------------------------------------------------------------------

# Larger λ → faster decay (half-life ≈ ln(2)/λ hours).
# llm    : half-life ≈ 14 h  (models drift / provider outages are common)
# search : half-life ≈ 24 h
# code   : half-life ≈ 48 h
# math   : half-life ≈ 5 days (very stable)
# storage: half-life ≈ 5 days
# default: half-life ≈ 24 h (unknown domains)
_DEFAULT_LAMBDA: Final[dict[str, float]] = {
    "llm":       0.049,   # ~14 h half-life
    "language":  0.049,
    "search":    0.029,   # ~24 h
    "code":      0.014,   # ~48 h
    "finance":   0.029,
    "legal":     0.014,
    "medical":   0.014,
    "vision":    0.029,
    "retrieval": 0.010,
    "math":      0.006,   # ~5 d
    "storage":   0.006,
    "translation": 0.010,
    "general":   0.029,   # default fallback
}


# ---------------------------------------------------------------------------
# VolatilityTable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityTable:
    """Mapping from tool domain to decay rate λ.

    Parameters
    ----------
    lambdas:
        ``{domain: λ}`` mapping.  Domains not listed fall back to
        ``default_lambda``.
    default_lambda:
        λ for unknown domains.  Default: ``0.029`` (~24 h half-life).

    Examples
    --------
    ::

        vt = VolatilityTable.default()
        print(vt.lambda_for("llm"))    # 0.049
        print(vt.lambda_for("math"))   # 0.006
    """

    lambdas: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_LAMBDA))
    default_lambda: float = 0.029

    @classmethod
    def default(cls) -> "VolatilityTable":
        """Return the built-in domain volatility table."""
        return cls(lambdas=dict(_DEFAULT_LAMBDA))

    def lambda_for(self, domain: str) -> float:
        """Return the λ decay coefficient for *domain*."""
        return self.lambdas.get(domain, self.default_lambda)

    def half_life_hours(self, domain: str) -> float:
        """Return the half-life in hours for *domain*'s decay rate."""
        lam = self.lambda_for(domain)
        if lam <= 0:
            return float("inf")
        return math.log(2) / lam


# ---------------------------------------------------------------------------
# ReputationDecay
# ---------------------------------------------------------------------------


@dataclass
class ReputationDecay:
    """Time-variant exponential reputation decay engine.

    Tracks the timestamp of the last signal for each tool and applies:

    .. math::

        Rep_t = Rep_{t-1} \\cdot e^{-\\lambda \\Delta t}

    where :math:`\\Delta t` is the age of the last signal **in hours**.

    Parameters
    ----------
    volatility:
        Domain → λ mapping.  Defaults to the built-in table.
    clock:
        Callable that returns the current POSIX timestamp in seconds.
        Override for testing (e.g. ``lambda: 1_000_000.0``).

    Example
    -------
    ::

        decay = ReputationDecay()
        decay.record_signal("gpt-4o", domain="llm", reliability=0.92)
        # … time passes …
        adjusted = decay.decayed_reliability("gpt-4o", base_reliability=0.92)
    """

    volatility: VolatilityTable = field(default_factory=VolatilityTable.default)
    clock: object = field(default=None, repr=False)

    # Internal state: tool_name → (last_signal_ts_seconds, domain, last_reliability)
    _signal_times: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _domains: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_reliability: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = time.time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal(
        self,
        tool_name: str,
        *,
        domain: str = "general",
        reliability: float = 1.0,
    ) -> None:
        """Record that a fresh signal was received for *tool_name*.

        Parameters
        ----------
        tool_name:
            Unique tool identifier.
        domain:
            Domain used to look up the volatility coefficient.
        reliability:
            The current (fresh) reliability score [0, 1].
        """
        now: float = self.clock()  # type: ignore[operator]
        self._signal_times[tool_name] = now
        self._domains[tool_name] = domain
        self._last_reliability[tool_name] = max(0.0, min(1.0, reliability))

    def age_hours(self, tool_name: str) -> float | None:
        """Return the age of the last signal for *tool_name* in hours.

        Returns ``None`` if no signal has been recorded.
        """
        ts = self._signal_times.get(tool_name)
        if ts is None:
            return None
        now: float = self.clock()  # type: ignore[operator]
        return max(0.0, (now - ts) / 3600.0)

    def decayed_reliability(
        self,
        tool_name: str,
        base_reliability: float,
        *,
        domain: str | None = None,
    ) -> float:
        """Return the time-decayed reliability for *tool_name*.

        If no signal has been recorded for *tool_name*, returns *base_reliability*
        unchanged.

        Parameters
        ----------
        tool_name:
            Unique tool identifier.
        base_reliability:
            The most recent un-decayed reliability score [0, 1].
        domain:
            Override for domain lookup.  Defaults to the domain recorded with
            the last signal, or ``"general"`` if unknown.

        Returns
        -------
        float
            Decayed reliability in [0, 1].
        """
        age = self.age_hours(tool_name)
        if age is None:
            return max(0.0, min(1.0, base_reliability))
        resolved_domain = domain or self._domains.get(tool_name, "general")
        lam = self.volatility.lambda_for(resolved_domain)
        decay_factor = math.exp(-lam * age)
        return max(0.0, min(1.0, base_reliability * decay_factor))

    def entropy_score(self, tool_name: str) -> float:
        """Return a [0, 1] entropy score representing how much reliability has decayed.

        0.0 = fresh signal (no decay), 1.0 = completely decayed.

        Parameters
        ----------
        tool_name:
            Unique tool identifier.

        Returns
        -------
        float
            Entropy in [0, 1].  Returns ``0.0`` if no signal has been recorded.
        """
        age = self.age_hours(tool_name)
        if age is None:
            return 0.0
        resolved_domain = self._domains.get(tool_name, "general")
        lam = self.volatility.lambda_for(resolved_domain)
        decay_factor = math.exp(-lam * age)
        return max(0.0, min(1.0, 1.0 - decay_factor))

    def system_entropy(self, tool_names: list[str] | None = None) -> float:
        """Return the mean entropy score across all (or specified) tracked tools.

        Parameters
        ----------
        tool_names:
            Tools to include.  Defaults to all tracked tools.

        Returns
        -------
        float
            Mean entropy in [0, 1].
        """
        names = tool_names if tool_names is not None else list(self._signal_times)
        if not names:
            return 0.0
        scores = [self.entropy_score(n) for n in names]
        return sum(scores) / len(scores)

    def all_tool_entropy(self) -> dict[str, float]:
        """Return ``{tool_name: entropy_score}`` for all tracked tools."""
        return {name: self.entropy_score(name) for name in self._signal_times}

    def tracked_tools(self) -> list[str]:
        """Return sorted list of all tool names with recorded signals."""
        return sorted(self._signal_times)
