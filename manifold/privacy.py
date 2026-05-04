"""Phase 35: Privacy Guard & Anonymisation — Differential Privacy for Federation.

Implements mathematical privacy primitives that protect MANIFOLD's outbound
gossip and threat intelligence feeds from "fingerprinting" attacks.

Two complementary mechanisms are provided:

1. **k-Anonymity Filter** — :class:`PrivacyGuard` groups outbound
   :class:`~manifold.federation.FederatedGossipPacket` or
   :class:`~manifold.threat_feed.ThreatIntelPayload` records by a shared
   quasi-identifier (``tool_name``).  A group is only released if it contains
   at least *k* distinct records, preventing re-identification of any single
   interaction.  Identifying fields (``org_id``, per-task context in
   ``details``) are stripped from surviving records.

2. **Lattice Noise (Differential Privacy)** — :meth:`PrivacyGuard.perturb_vector`
   adds zero-mean Laplacian noise to a numeric vector before federation::

       V_pub = V_actual + Laplace(0, Δf / ε)

   This ensures the "gist" of reliability is shared while exact proprietary
   performance metrics remain private.  The scale parameter ``b = Δf / ε``
   grows as the privacy budget *ε* shrinks.

Key classes
-----------
``PrivacyGuard``
    Combined k-anonymity filter and Laplacian noise engine.
``PrivacyConfig``
    Immutable configuration snapshot for a :class:`PrivacyGuard`.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any

from .federation import FederatedGossipPacket
from .threat_feed import ThreatIntelPayload


# ---------------------------------------------------------------------------
# PrivacyConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyConfig:
    """Immutable configuration for :class:`PrivacyGuard`.

    Attributes
    ----------
    k:
        Minimum group size required for k-anonymity.  Groups smaller than
        *k* are suppressed.  Must be >= 1.
    epsilon:
        Differential-privacy budget (ε).  Smaller values mean more noise
        and stronger privacy.  Must be > 0.
    sensitivity:
        Global sensitivity (Δf) of the vector function being perturbed.
        Defaults to ``1.0``.
    seed:
        Optional RNG seed for reproducibility in tests.  ``None`` for a
        non-deterministic sequence.
    """

    k: int = 5
    epsilon: float = 1.0
    sensitivity: float = 1.0
    seed: int | None = None


# ---------------------------------------------------------------------------
# PrivacyGuard
# ---------------------------------------------------------------------------


@dataclass
class PrivacyGuard:
    """Combined k-Anonymity filter and Laplacian differential-privacy engine.

    Parameters
    ----------
    k:
        Minimum group size for k-anonymity.  Defaults to ``5``.
    epsilon:
        Differential-privacy budget (ε > 0).  Defaults to ``1.0``.
    sensitivity:
        Global sensitivity Δf for the :meth:`perturb_vector` Laplace scale.
        Defaults to ``1.0``.
    seed:
        Optional RNG seed.

    Example
    -------
    ::

        guard = PrivacyGuard(k=3, epsilon=0.5)

        # k-anonymity: suppress groups with < 3 members
        safe_packets = guard.filter_gossip(packets)

        # Differential privacy: add Laplace noise to a score vector
        noisy_scores = guard.perturb_vector([0.9, 0.8, 0.75])
    """

    k: int = 5
    epsilon: float = 1.0
    sensitivity: float = 1.0
    seed: int | None = None

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1; got {self.k!r}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0; got {self.epsilon!r}")
        if self.sensitivity <= 0.0:
            raise ValueError(f"sensitivity must be > 0; got {self.sensitivity!r}")
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # k-Anonymity: gossip packets
    # ------------------------------------------------------------------

    def filter_gossip(
        self,
        packets: list[FederatedGossipPacket],
    ) -> list[FederatedGossipPacket]:
        """Apply k-Anonymity to a list of :class:`~manifold.federation.FederatedGossipPacket` records.

        Records are grouped by ``tool_name``.  Any group with fewer than
        *k* members is suppressed entirely.  Surviving records have their
        ``org_id`` replaced with ``"anonymous"`` to prevent org-level
        fingerprinting.

        Parameters
        ----------
        packets:
            Input gossip packets.

        Returns
        -------
        list[FederatedGossipPacket]
            Anonymised packets satisfying k-anonymity.
        """
        groups: dict[str, list[FederatedGossipPacket]] = {}
        for p in packets:
            groups.setdefault(p.tool_name, []).append(p)

        result: list[FederatedGossipPacket] = []
        for group in groups.values():
            if len(group) < self.k:
                continue
            for p in group:
                result.append(
                    FederatedGossipPacket(
                        tool_name=p.tool_name,
                        signal=p.signal,
                        confidence=p.confidence,
                        org_id="anonymous",
                        weight=p.weight,
                    )
                )
        return result

    # ------------------------------------------------------------------
    # k-Anonymity: threat intel payloads
    # ------------------------------------------------------------------

    def filter_threat(
        self,
        payloads: list[ThreatIntelPayload],
    ) -> list[ThreatIntelPayload]:
        """Apply k-Anonymity to a list of :class:`~manifold.threat_feed.ThreatIntelPayload` records.

        Records are grouped by ``(tool_name, event_type)``.  Groups with
        fewer than *k* members are suppressed.  Surviving records have
        per-task context (``task_id``, ``org_id``, ``prompt_context``)
        stripped from the ``details`` field, and ``source`` is normalised
        to ``"manifold"``.

        Parameters
        ----------
        payloads:
            Input threat intel payloads.

        Returns
        -------
        list[ThreatIntelPayload]
            Anonymised payloads satisfying k-anonymity.
        """
        _STRIP_KEYS = frozenset({"task_id", "org_id", "prompt_context", "session_id"})

        groups: dict[tuple[str, str], list[ThreatIntelPayload]] = {}
        for p in payloads:
            key = (p.tool_name, p.event_type)
            groups.setdefault(key, []).append(p)

        result: list[ThreatIntelPayload] = []
        for group in groups.values():
            if len(group) < self.k:
                continue
            for p in group:
                clean_details: dict[str, Any] = {
                    k: v for k, v in p.details.items() if k not in _STRIP_KEYS
                }
                result.append(
                    ThreatIntelPayload(
                        event_type=p.event_type,
                        tool_name=p.tool_name,
                        severity=p.severity,
                        timestamp=p.timestamp,
                        details=clean_details,
                        source="manifold",
                    )
                )
        return result

    # ------------------------------------------------------------------
    # Differential privacy: Laplacian vector perturbation
    # ------------------------------------------------------------------

    def perturb_vector(
        self,
        values: list[float],
        sensitivity: float | None = None,
    ) -> list[float]:
        """Add independent Laplacian noise to each element of *values*.

        The scale parameter is ``b = sensitivity / epsilon``.  Each element
        receives an independent draw from ``Laplace(0, b)``.

        Parameters
        ----------
        values:
            Numeric vector to perturb (e.g. 4-grid reliability scores).
        sensitivity:
            Global sensitivity Δf override.  If ``None``, uses
            ``self.sensitivity``.

        Returns
        -------
        list[float]
            Perturbed vector ``V_pub = V_actual + Laplace(0, Δf/ε)``.
        """
        b = (sensitivity if sensitivity is not None else self.sensitivity) / self.epsilon
        with self._lock:
            return [v + self._sample_laplace(b) for v in values]

    # ------------------------------------------------------------------
    # Configuration snapshot
    # ------------------------------------------------------------------

    def config(self) -> PrivacyConfig:
        """Return an immutable :class:`PrivacyConfig` snapshot."""
        return PrivacyConfig(
            k=self.k,
            epsilon=self.epsilon,
            sensitivity=self.sensitivity,
            seed=self.seed,
        )

    def summary(self) -> dict[str, object]:
        """Return a summary dict of the current privacy configuration."""
        return {
            "k": self.k,
            "epsilon": self.epsilon,
            "sensitivity": self.sensitivity,
            "laplace_scale": self.sensitivity / self.epsilon,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sample_laplace(self, b: float) -> float:
        """Draw one sample from Laplace(0, b) using the inverse CDF method.

        For ``U ~ Uniform(0, 1)`` (exclusive of 0 and 1)::

            X = -b · sign(U - 0.5) · ln(1 - 2|U - 0.5|)

        This avoids external dependencies while being numerically stable.
        """
        u = self._rng.random()
        # Guard against the (near-impossible) degenerate boundary values
        while u == 0.0 or u == 1.0:
            u = self._rng.random()
        half_diff = abs(u - 0.5)
        sign = 1.0 if u >= 0.5 else -1.0
        return -b * sign * math.log(1.0 - 2.0 * half_diff)
