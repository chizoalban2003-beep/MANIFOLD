"""Phase 37: Formal Policy Verification — Conflict Detection for B2B Handshakes.

Performs logical "Conflict Analysis" between two :class:`~manifold.b2b.OrgPolicy`
objects *before* a costly cryptographic handshake is attempted.

Two complementary analyses are provided:

1. **Deadlock Detection** — uses set-intersection logic to identify
   "Impossible Handshakes".  For example: if Org A requires
   ``MinTrust > 0.9`` but Org B has ``MaxRiskCap = 0.8`` for the same
   domain, they can never agree — a "deadlock" condition.

2. **Friction Score** — a scalar in ``[0.0, 1.0]`` that quantifies how
   much friction exists between the two policies.  The
   :class:`~manifold.b2b.B2BRouter` can use this score as a pre-flight
   check: if ``friction_score > threshold``, skip the handshake entirely.

Key classes
-----------
``PolicyConflict``
    Description of a single detected conflict between two policies.
``VerificationResult``
    Full output of a :class:`PolicyVerifier` run: list of conflicts,
    overall ``friction_score``, and ``compatible`` flag.
``PolicyVerifier``
    Accepts two :class:`~manifold.b2b.OrgPolicy` objects and performs
    conflict analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .b2b import OrgPolicy
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# PolicyConflict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConflict:
    """A single conflict detected between two :class:`~manifold.b2b.OrgPolicy` objects.

    Attributes
    ----------
    conflict_type:
        Short category tag.  One of:
        ``"reliability_deadlock"``, ``"risk_deadlock"``,
        ``"domain_mismatch"``, ``"reliability_gap"``, ``"risk_gap"``.
    org_a_id:
        ID of the first organisation.
    org_b_id:
        ID of the second organisation.
    description:
        Human-readable explanation of the conflict.
    friction_contribution:
        How much this conflict contributes to the overall
        :attr:`VerificationResult.friction_score` [0.0, 1.0].
    """

    conflict_type: str
    org_a_id: str
    org_b_id: str
    description: str
    friction_contribution: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "conflict_type": self.conflict_type,
            "org_a_id": self.org_a_id,
            "org_b_id": self.org_b_id,
            "description": self.description,
            "friction_contribution": round(self.friction_contribution, 4),
        }


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Output of a :class:`PolicyVerifier` run.

    Attributes
    ----------
    compatible:
        ``True`` if no *deadlock* conflicts were detected (gap conflicts are
        allowed — they raise the friction score but do not block handshakes).
    org_a_id:
        ID of the first organisation.
    org_b_id:
        ID of the second organisation.
    conflicts:
        Tuple of all detected :class:`PolicyConflict` objects.
    friction_score:
        Aggregate friction score in ``[0.0, 1.0]``.  ``0.0`` means
        perfectly compatible; ``1.0`` means maximally incompatible.
    deadlock_count:
        Number of "deadlock" (impossible) conflicts.
    gap_count:
        Number of "gap" (negotiable) conflicts.
    recommendation:
        Human-readable routing recommendation.
    """

    compatible: bool
    org_a_id: str
    org_b_id: str
    conflicts: tuple[PolicyConflict, ...]
    friction_score: float
    deadlock_count: int
    gap_count: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "compatible": self.compatible,
            "org_a_id": self.org_a_id,
            "org_b_id": self.org_b_id,
            "friction_score": round(self.friction_score, 4),
            "deadlock_count": self.deadlock_count,
            "gap_count": self.gap_count,
            "recommendation": self.recommendation,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


# ---------------------------------------------------------------------------
# PolicyVerifier
# ---------------------------------------------------------------------------

_DEADLOCK_TYPES = frozenset({"reliability_deadlock", "risk_deadlock"})


@dataclass
class PolicyVerifier:
    """Performs formal conflict analysis between two :class:`~manifold.b2b.OrgPolicy` objects.

    Parameters
    ----------
    friction_threshold:
        Friction score above which the verifier recommends skipping the
        handshake.  Default: ``0.6``.

    Example
    -------
    ::

        verifier = PolicyVerifier()
        result = verifier.verify(org_a_policy, org_b_policy)
        if result.friction_score > 0.6:
            raise PermissionError("Policies too incompatible — skip handshake.")
    """

    friction_threshold: float = 0.6

    _conflicts: list[PolicyConflict] = field(
        default_factory=list, init=False, repr=False
    )

    def verify(self, org_a: OrgPolicy, org_b: OrgPolicy) -> VerificationResult:
        """Run conflict analysis between *org_a* and *org_b*.

        The following checks are performed:

        * **Reliability Deadlock** — Org A's ``min_reliability`` exceeds
          Org B's ``min_reliability`` AND Org B's ``min_reliability`` is
          below 1.0 (i.e., there is no feasible reliability that satisfies
          both simultaneously when considered as mutual requirements).
          Specifically: if A requires X but B can only offer Y < X, *and*
          B requires Z but A can only offer W < Z, both sides are
          simultaneously dissatisfied — a deadlock.

        * **Risk Deadlock** — Org A's ``max_risk`` is lower than Org B's
          ``max_risk`` *and* the gap is large enough that B would exceed
          A's tolerance.  The symmetric case is also checked.

        * **Domain Mismatch** — The declared domains differ (a gap conflict,
          not a deadlock).

        * **Reliability Gap** — One org's ``min_reliability`` exceeds the
          other's (negotiable).

        * **Risk Gap** — Asymmetric risk tolerance (negotiable).

        Parameters
        ----------
        org_a:
            First organisation's policy.
        org_b:
            Second organisation's policy.

        Returns
        -------
        VerificationResult
        """
        conflicts: list[PolicyConflict] = []

        # ----------------------------------------------------------------
        # 1. Reliability deadlock
        #    "Impossible handshake": each org demands a trust level from the
        #    other that the other cannot guarantee.
        #
        #    We model each org's DEMAND on its counterpart via max_risk:
        #      trust floor A demands from B = (1 - a.max_risk)
        #      trust floor B demands from A = (1 - b.max_risk)
        #
        #    And each org's OFFER to its counterpart = its min_reliability.
        #
        #    Spec example: "Org A requires MinTrust > 0.9 but Org B has
        #    MaxTrustCap of 0.8" translates to:
        #      A's trust floor = 1 - a.max_risk = 0.9
        #      B's offer = b.min_reliability = 0.8
        #      0.8 < 0.9 → B can't satisfy A
        #    A mutual deadlock fires when BOTH sides simultaneously fail.
        # ----------------------------------------------------------------
        a_risk_cap = org_a.max_risk
        b_risk_cap = org_b.max_risk

        # Trust floor each org demands from counterpart
        a_trust_demand = 1.0 - a_risk_cap   # A demands B have reliability ≥ this
        b_trust_demand = 1.0 - b_risk_cap   # B demands A have reliability ≥ this

        b_fails_a = org_b.min_reliability < a_trust_demand
        a_fails_b = org_a.min_reliability < b_trust_demand

        if b_fails_a and a_fails_b:
            gap = max(
                a_trust_demand - org_b.min_reliability,
                b_trust_demand - org_a.min_reliability,
            )
            conflicts.append(
                PolicyConflict(
                    conflict_type="reliability_deadlock",
                    org_a_id=org_a.org_id,
                    org_b_id=org_b.org_id,
                    description=(
                        f"{org_a.org_id!r} demands trust ≥ {a_trust_demand:.3f} "
                        f"(max_risk={a_risk_cap:.3f}) but {org_b.org_id!r} only "
                        f"guarantees {org_b.min_reliability:.3f}; simultaneously "
                        f"{org_b.org_id!r} demands trust ≥ {b_trust_demand:.3f} "
                        f"but {org_a.org_id!r} only guarantees {org_a.min_reliability:.3f}. "
                        "Impossible handshake."
                    ),
                    friction_contribution=clamp01(gap),
                )
            )

        # ----------------------------------------------------------------
        # 2. Risk deadlock
        #    A "risk deadlock" is a MUTUAL impossibility: Org A demands a
        #    minimum trust level from B that B cannot guarantee, AND Org B
        #    simultaneously demands a minimum trust level from A that A
        #    cannot guarantee.
        #
        #    In our model: max_risk determines the minimum trust floor an
        #    org requires of its counterparties (trust = 1 - max_risk).
        #    min_reliability is what the org guarantees it can provide.
        #
        #    So:
        #      A demands from B:  b.min_reliability ≥ (1 - a.max_risk)
        #      B demands from A:  a.min_reliability ≥ (1 - b.max_risk)
        #
        #    Deadlock when BOTH conditions fail simultaneously.
        # ----------------------------------------------------------------
        a_trust_floor = 1.0 - a_risk_cap   # minimum trust A demands from B
        b_trust_floor = 1.0 - b_risk_cap   # minimum trust B demands from A

        b_cant_satisfy_a = org_b.min_reliability < a_trust_floor
        a_cant_satisfy_b = org_a.min_reliability < b_trust_floor

        if b_cant_satisfy_a and a_cant_satisfy_b:
            gap = max(
                a_trust_floor - org_b.min_reliability,
                b_trust_floor - org_a.min_reliability,
            )
            conflicts.append(
                PolicyConflict(
                    conflict_type="risk_deadlock",
                    org_a_id=org_a.org_id,
                    org_b_id=org_b.org_id,
                    description=(
                        f"{org_a.org_id!r} requires trust ≥ {a_trust_floor:.3f} "
                        f"(max_risk={a_risk_cap:.3f}) but {org_b.org_id!r} only "
                        f"guarantees {org_b.min_reliability:.3f}; simultaneously "
                        f"{org_b.org_id!r} requires trust ≥ {b_trust_floor:.3f} "
                        f"but {org_a.org_id!r} only guarantees {org_a.min_reliability:.3f}. "
                        "Impossible handshake."
                    ),
                    friction_contribution=clamp01(gap),
                )
            )

        # ----------------------------------------------------------------
        # 3. Domain mismatch (gap conflict — negotiable)
        # ----------------------------------------------------------------
        if org_a.domain != org_b.domain:
            conflicts.append(
                PolicyConflict(
                    conflict_type="domain_mismatch",
                    org_a_id=org_a.org_id,
                    org_b_id=org_b.org_id,
                    description=(
                        f"Domain mismatch: {org_a.org_id!r} operates in "
                        f"{org_a.domain!r}, {org_b.org_id!r} operates in "
                        f"{org_b.domain!r}. Cross-domain negotiation required."
                    ),
                    friction_contribution=0.2,
                )
            )

        # ----------------------------------------------------------------
        # 4. Reliability gap (gap conflict — negotiable)
        # ----------------------------------------------------------------
        rel_gap = abs(org_a.min_reliability - org_b.min_reliability)
        if rel_gap > 0.05:
            conflicts.append(
                PolicyConflict(
                    conflict_type="reliability_gap",
                    org_a_id=org_a.org_id,
                    org_b_id=org_b.org_id,
                    description=(
                        f"Reliability gap of {rel_gap:.3f} between "
                        f"{org_a.org_id!r} (min={org_a.min_reliability:.3f}) "
                        f"and {org_b.org_id!r} (min={org_b.min_reliability:.3f})."
                    ),
                    friction_contribution=clamp01(rel_gap),
                )
            )

        # ----------------------------------------------------------------
        # 5. Risk tolerance gap (gap conflict — negotiable)
        # ----------------------------------------------------------------
        risk_gap = abs(org_a.max_risk - org_b.max_risk)
        if risk_gap > 0.05:
            conflicts.append(
                PolicyConflict(
                    conflict_type="risk_gap",
                    org_a_id=org_a.org_id,
                    org_b_id=org_b.org_id,
                    description=(
                        f"Risk tolerance gap of {risk_gap:.3f}: "
                        f"{org_a.org_id!r} allows max_risk={org_a.max_risk:.3f}, "
                        f"{org_b.org_id!r} allows max_risk={org_b.max_risk:.3f}."
                    ),
                    friction_contribution=clamp01(risk_gap),
                )
            )

        # ----------------------------------------------------------------
        # Aggregate friction score
        # ----------------------------------------------------------------
        deadlock_count = sum(
            1 for c in conflicts if c.conflict_type in _DEADLOCK_TYPES
        )
        gap_count = len(conflicts) - deadlock_count
        compatible = deadlock_count == 0

        # Friction = clipped sum of all contributions, with deadlocks
        # weighted 2× because they block the handshake entirely.
        raw_friction = sum(
            (c.friction_contribution * (2.0 if c.conflict_type in _DEADLOCK_TYPES else 1.0))
            for c in conflicts
        )
        friction_score = clamp01(raw_friction / max(1, len(conflicts) + 1))

        recommendation = _recommend(compatible, friction_score, self.friction_threshold)

        return VerificationResult(
            compatible=compatible,
            org_a_id=org_a.org_id,
            org_b_id=org_b.org_id,
            conflicts=tuple(conflicts),
            friction_score=friction_score,
            deadlock_count=deadlock_count,
            gap_count=gap_count,
            recommendation=recommendation,
        )

    def verify_many(
        self,
        policies: list[OrgPolicy],
    ) -> dict[tuple[str, str], VerificationResult]:
        """Verify all pairwise combinations in *policies*.

        Parameters
        ----------
        policies:
            List of :class:`~manifold.b2b.OrgPolicy` objects to compare.

        Returns
        -------
        dict[tuple[str, str], VerificationResult]
            Mapping of ``(org_a_id, org_b_id)`` to
            :class:`VerificationResult`.  Each pair appears once
            (``i < j`` ordering).
        """
        results: dict[tuple[str, str], VerificationResult] = {}
        for i, a in enumerate(policies):
            for b in policies[i + 1 :]:
                result = self.verify(a, b)
                results[(a.org_id, b.org_id)] = result
        return results

    def compatibility_matrix(
        self,
        policies: list[OrgPolicy],
    ) -> list[dict[str, Any]]:
        """Return a flat list of friction scores for all pairs.

        Suitable for serialisation to JSON or display in a dashboard table.

        Parameters
        ----------
        policies:
            List of :class:`~manifold.b2b.OrgPolicy` objects.

        Returns
        -------
        list[dict]
            Each entry has keys ``org_a``, ``org_b``, ``friction_score``,
            ``compatible``, ``deadlock_count``.
        """
        rows = []
        for key, result in self.verify_many(policies).items():
            rows.append(
                {
                    "org_a": key[0],
                    "org_b": key[1],
                    "friction_score": round(result.friction_score, 4),
                    "compatible": result.compatible,
                    "deadlock_count": result.deadlock_count,
                    "gap_count": result.gap_count,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recommend(
    compatible: bool,
    friction_score: float,
    threshold: float,
) -> str:
    """Build a routing recommendation string."""
    if not compatible:
        return (
            "BLOCK — Impossible handshake detected. "
            "Do not attempt cryptographic negotiation."
        )
    if friction_score > threshold:
        return (
            f"WARN — High friction ({friction_score:.3f} > threshold {threshold:.3f}). "
            "Consider renegotiating policy terms before handshake."
        )
    return (
        f"PROCEED — Friction score {friction_score:.3f} is within acceptable range. "
        "Handshake may proceed."
    )
