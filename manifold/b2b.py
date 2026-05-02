"""Phase 20: Inter-Organisational B2B Routing.

As organisations adopt MANIFOLD, they need a way to safely interact with
*each other's* agents.  Phase 20 introduces cross-org routing that uses
``ManifoldPolicy`` (Phase 18) and the ``ReputationHub`` (Phase 15) as the
foundation for trust-gated agent-to-agent API calls.

Key ideas
---------
* **Policy Handshake** — before an agent from Org A calls a service from
  Org B, their MANIFOLD instances exchange ``ManifoldPolicy`` summaries.
  If Org A's risk tolerance is stricter than Org B's declared reliability,
  the call is blocked.

* **B2B Router** — routes cross-org requests through a compatibility check,
  returning a structured ``B2BRouteResult`` that tells the caller whether
  to proceed, negotiate, or block.

* **Agent Economy Ledger** — tracks micro-transaction "costs" for each
  cross-org call, denominated in trust units derived from ``ReputationHub``
  scores.  Low-reputation callers pay a higher surcharge.

Key classes
-----------
``OrgPolicy``
    Lightweight snapshot of another organisation's published policy.
``PolicyHandshake``
    Compares the local ``ManifoldPolicy`` with a remote ``OrgPolicy`` and
    decides whether the call is compatible.
``HandshakeResult``
    Outcome of a ``PolicyHandshake`` check.
``B2BRouteResult``
    Full routing decision for a cross-org call.
``B2BRouter``
    Orchestrates handshake + reputation check + economy ledger.
``AgentEconomyLedger``
    Records micro-transaction trust costs for every cross-org call.
``EconomyEntry``
    A single ledger entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .hub import ReputationHub
from .policy import ManifoldPolicy, PolicyDomain
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# OrgPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrgPolicy:
    """A lightweight snapshot of a remote organisation's published policy.

    Attributes
    ----------
    org_id:
        Unique identifier for the remote organisation.
    min_reliability:
        The minimum tool reliability this org *guarantees* for its agents.
    max_risk:
        The maximum risk score any call from this org may produce.
    domain:
        Primary operational domain of the remote org (e.g. ``"finance"``).
    version:
        Policy version string.
    notes:
        Free-text notes published alongside the policy.

    Example
    -------
    ::

        remote = OrgPolicy(
            org_id="org-b",
            min_reliability=0.85,
            max_risk=0.30,
            domain="finance",
        )
    """

    org_id: str
    min_reliability: float = 0.75
    max_risk: float = 0.40
    domain: str = "general"
    version: str = "1.0.0"
    notes: str = ""

    @classmethod
    def from_manifold_policy(
        cls, policy: ManifoldPolicy, org_id: str
    ) -> "OrgPolicy":
        """Build an ``OrgPolicy`` snapshot from a full ``ManifoldPolicy``.

        Parameters
        ----------
        policy:
            The local ``ManifoldPolicy`` to summarise.
        org_id:
            Identifier for this organisation.

        Returns
        -------
        OrgPolicy
        """
        if policy.domains:
            domain_cfg = policy.domains[0]
        else:
            domain_cfg = PolicyDomain(name="general")
        return cls(
            org_id=org_id,
            min_reliability=domain_cfg.min_tool_reliability,
            max_risk=domain_cfg.risk_tolerance,
            domain=domain_cfg.name,
            version=policy.version,
            notes=domain_cfg.notes,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "org_id": self.org_id,
            "min_reliability": self.min_reliability,
            "max_risk": self.max_risk,
            "domain": self.domain,
            "version": self.version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OrgPolicy":
        """Deserialise from a plain dict."""
        return cls(
            org_id=str(data.get("org_id", "unknown")),
            min_reliability=float(data.get("min_reliability", 0.75)),  # type: ignore[arg-type]
            max_risk=float(data.get("max_risk", 0.40)),  # type: ignore[arg-type]
            domain=str(data.get("domain", "general")),
            version=str(data.get("version", "1.0.0")),
            notes=str(data.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# PolicyHandshake
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandshakeResult:
    """Outcome of a ``PolicyHandshake`` between two organisations.

    Attributes
    ----------
    compatible:
        ``True`` if the policies are compatible and the call may proceed.
    local_org_id:
        ID of the calling organisation.
    remote_org_id:
        ID of the target organisation.
    local_domain:
        Domain the local policy is configured for.
    remote_domain:
        Domain the remote org declared.
    conflict_reasons:
        Human-readable reasons for incompatibility (empty when compatible).
    risk_delta:
        ``remote.max_risk - local.risk_tolerance`` — positive means remote
        allows more risk than local tolerates.
    reliability_delta:
        ``remote.min_reliability - local.min_reliability`` — positive means
        remote offers better reliability than local requires.
    """

    compatible: bool
    local_org_id: str
    remote_org_id: str
    local_domain: str
    remote_domain: str
    conflict_reasons: tuple[str, ...]
    risk_delta: float
    reliability_delta: float


class PolicyHandshake:
    """Compares the local ``ManifoldPolicy`` with a remote ``OrgPolicy``.

    The handshake checks two conditions:

    1. **Risk Gate** — the remote org's ``max_risk`` must be ≤ the local
       org's configured ``risk_tolerance`` for the relevant domain.
    2. **Reliability Gate** — the remote org's ``min_reliability`` must be
       ≥ the local org's ``min_tool_reliability`` requirement.

    Parameters
    ----------
    local_policy:
        The local ``ManifoldPolicy``.
    local_org_id:
        Identifier for the local organisation.

    Example
    -------
    ::

        handshake = PolicyHandshake(local_policy=policy, local_org_id="org-a")
        result = handshake.check(remote_policy)
        if not result.compatible:
            raise PermissionError(result.conflict_reasons)
    """

    def __init__(self, local_policy: ManifoldPolicy, local_org_id: str = "local") -> None:
        self.local_policy = local_policy
        self.local_org_id = local_org_id

    def check(self, remote: OrgPolicy, domain: str | None = None) -> HandshakeResult:
        """Check compatibility with a remote ``OrgPolicy``.

        Parameters
        ----------
        remote:
            Remote organisation's policy snapshot.
        domain:
            Domain context.  If ``None``, uses the remote org's declared
            domain.

        Returns
        -------
        HandshakeResult
        """
        effective_domain = domain or remote.domain
        local_domain_cfg = self.local_policy.domain(effective_domain)
        if local_domain_cfg is None:
            # Fall back to global settings
            local_risk_tol = self.local_policy.global_veto_threshold
            local_min_rel = 0.70
            local_domain = "general"
        else:
            local_risk_tol = local_domain_cfg.risk_tolerance
            local_min_rel = local_domain_cfg.min_tool_reliability
            local_domain = local_domain_cfg.name

        conflicts: list[str] = []

        # Risk gate: remote max_risk must not exceed local tolerance
        if remote.max_risk > local_risk_tol:
            conflicts.append(
                f"Remote max_risk={remote.max_risk:.3f} exceeds local "
                f"risk_tolerance={local_risk_tol:.3f} for domain={effective_domain!r}"
            )

        # Reliability gate: remote min_reliability must meet local minimum
        if remote.min_reliability < local_min_rel:
            conflicts.append(
                f"Remote min_reliability={remote.min_reliability:.3f} is below "
                f"local requirement={local_min_rel:.3f} for domain={effective_domain!r}"
            )

        risk_delta = remote.max_risk - local_risk_tol
        rel_delta = remote.min_reliability - local_min_rel

        return HandshakeResult(
            compatible=len(conflicts) == 0,
            local_org_id=self.local_org_id,
            remote_org_id=remote.org_id,
            local_domain=local_domain,
            remote_domain=remote.domain,
            conflict_reasons=tuple(conflicts),
            risk_delta=risk_delta,
            reliability_delta=rel_delta,
        )


# ---------------------------------------------------------------------------
# B2BRouteResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B2BRouteResult:
    """Full routing decision for a cross-org call.

    Attributes
    ----------
    allowed:
        ``True`` if the call may proceed.
    local_org_id:
        Calling org.
    remote_org_id:
        Target org.
    handshake:
        Policy handshake result.
    reputation_score:
        Reputation score of the remote org from ``ReputationHub`` (0–1).
    surcharge:
        Trust surcharge applied to the call cost (higher for low-rep orgs).
    net_trust_cost:
        ``base_cost * (1 + surcharge)`` — the cost in trust units.
    block_reason:
        Human-readable reason for blocking, or ``""`` if allowed.
    """

    allowed: bool
    local_org_id: str
    remote_org_id: str
    handshake: HandshakeResult
    reputation_score: float
    surcharge: float
    net_trust_cost: float
    block_reason: str


# ---------------------------------------------------------------------------
# B2BRouter
# ---------------------------------------------------------------------------


@dataclass
class B2BRouter:
    """Orchestrates cross-org routing with policy handshakes and reputation checks.

    Parameters
    ----------
    local_policy:
        The calling organisation's ``ManifoldPolicy``.
    hub:
        ``ReputationHub`` instance used to look up remote-org reputation.
    local_org_id:
        Identifier for the calling organisation.
    base_cost:
        Base cost per cross-org call in trust units.  Default: ``1.0``.
    min_reputation:
        Minimum remote-org reputation to allow a call.  Default: ``0.5``.
    surcharge_exponent:
        Exponential factor controlling how fast surcharge grows as reputation
        drops.  Default: ``2.0``.

    Example
    -------
    ::

        router = B2BRouter(local_policy=policy, hub=hub, local_org_id="org-a")
        result = router.route(remote_policy=remote_org_policy, remote_org_id="org-b")
        if result.allowed:
            # ... make the API call
            router.ledger.record(result)
    """

    local_policy: ManifoldPolicy
    hub: ReputationHub
    local_org_id: str = "local"
    base_cost: float = 1.0
    min_reputation: float = 0.5
    surcharge_exponent: float = 2.0

    ledger: "AgentEconomyLedger" = field(init=False)
    _handshake: PolicyHandshake = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ledger = AgentEconomyLedger()
        self._handshake = PolicyHandshake(
            local_policy=self.local_policy,
            local_org_id=self.local_org_id,
        )

    def route(
        self,
        remote: OrgPolicy,
        domain: str | None = None,
        auto_record: bool = True,
    ) -> B2BRouteResult:
        """Evaluate a cross-org routing request.

        Parameters
        ----------
        remote:
            The remote organisation's policy snapshot.
        domain:
            Domain context override.
        auto_record:
            If ``True``, automatically record the result in the ``ledger``.

        Returns
        -------
        B2BRouteResult
        """
        handshake = self._handshake.check(remote, domain=domain)

        # Reputation lookup — use the org_id as the tool name in the hub
        rep_score = self._reputation(remote.org_id)
        surcharge = self._surcharge(rep_score)
        net_cost = self.base_cost * (1.0 + surcharge)

        block_reason = ""
        if not handshake.compatible:
            block_reason = "; ".join(handshake.conflict_reasons)
        elif rep_score < self.min_reputation:
            block_reason = (
                f"Remote org {remote.org_id!r} reputation={rep_score:.3f} "
                f"< min_reputation={self.min_reputation:.3f}"
            )

        allowed = not block_reason

        result = B2BRouteResult(
            allowed=allowed,
            local_org_id=self.local_org_id,
            remote_org_id=remote.org_id,
            handshake=handshake,
            reputation_score=rep_score,
            surcharge=surcharge,
            net_trust_cost=net_cost,
            block_reason=block_reason,
        )

        if auto_record:
            self.ledger.record(result)

        return result

    def route_from_policy(
        self,
        remote_policy: ManifoldPolicy,
        remote_org_id: str,
        domain: str | None = None,
        auto_record: bool = True,
    ) -> B2BRouteResult:
        """Convenience: build an ``OrgPolicy`` from a full ``ManifoldPolicy``.

        Parameters
        ----------
        remote_policy:
            The remote org's full ``ManifoldPolicy``.
        remote_org_id:
            Identifier for the remote organisation.
        domain:
            Domain context override.
        auto_record:
            Auto-record in ledger.

        Returns
        -------
        B2BRouteResult
        """
        remote = OrgPolicy.from_manifold_policy(remote_policy, org_id=remote_org_id)
        return self.route(remote, domain=domain, auto_record=auto_record)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reputation(self, org_id: str) -> float:
        """Look up remote org reputation from the hub, defaulting to 0.75."""
        try:
            score = self.hub.live_reliability(org_id)
            if score is None:
                return 0.75
            return clamp01(score)
        except (KeyError, ZeroDivisionError):
            return 0.75  # neutral default if no data

    def _surcharge(self, reputation: float) -> float:
        """Compute trust surcharge: higher surcharge for lower reputation.

        ``surcharge = (1 - reputation) ** exponent``
        """
        return clamp01((1.0 - reputation) ** self.surcharge_exponent)

    def summary(self) -> dict[str, object]:
        """Return a summary of all routing decisions."""
        return self.ledger.summary()


# ---------------------------------------------------------------------------
# AgentEconomyLedger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EconomyEntry:
    """A single entry in the ``AgentEconomyLedger``.

    Attributes
    ----------
    local_org_id:
        Calling org.
    remote_org_id:
        Target org.
    allowed:
        Whether the call was allowed.
    reputation_score:
        Reputation of the remote org at call time.
    surcharge:
        Trust surcharge applied.
    net_trust_cost:
        Total cost in trust units.
    block_reason:
        Block reason if blocked, else ``""``.
    """

    local_org_id: str
    remote_org_id: str
    allowed: bool
    reputation_score: float
    surcharge: float
    net_trust_cost: float
    block_reason: str

    @classmethod
    def from_route_result(cls, result: B2BRouteResult) -> "EconomyEntry":
        """Build an entry from a ``B2BRouteResult``."""
        return cls(
            local_org_id=result.local_org_id,
            remote_org_id=result.remote_org_id,
            allowed=result.allowed,
            reputation_score=result.reputation_score,
            surcharge=result.surcharge,
            net_trust_cost=result.net_trust_cost,
            block_reason=result.block_reason,
        )


@dataclass
class AgentEconomyLedger:
    """Records micro-transaction trust costs for every cross-org call.

    The ledger accumulates ``EconomyEntry`` objects and provides analytics
    over the full call history.

    Example
    -------
    ::

        ledger = AgentEconomyLedger()
        ledger.record(route_result)
        print(ledger.total_trust_cost())
        print(ledger.block_rate())
    """

    _entries: list[EconomyEntry] = field(default_factory=list, init=False, repr=False)

    def record(self, result: B2BRouteResult) -> None:
        """Record a routing result.

        Parameters
        ----------
        result:
            The ``B2BRouteResult`` to add to the ledger.
        """
        self._entries.append(EconomyEntry.from_route_result(result))

    def entries(self) -> list[EconomyEntry]:
        """Return all ledger entries."""
        return list(self._entries)

    def total_trust_cost(self) -> float:
        """Return total trust cost across all calls."""
        return sum(e.net_trust_cost for e in self._entries)

    def allowed_count(self) -> int:
        """Return number of allowed calls."""
        return sum(1 for e in self._entries if e.allowed)

    def blocked_count(self) -> int:
        """Return number of blocked calls."""
        return sum(1 for e in self._entries if not e.allowed)

    def block_rate(self) -> float:
        """Return fraction of calls that were blocked [0, 1]."""
        total = len(self._entries)
        return self.blocked_count() / total if total > 0 else 0.0

    def avg_surcharge(self) -> float:
        """Return the average trust surcharge across all calls."""
        if not self._entries:
            return 0.0
        return sum(e.surcharge for e in self._entries) / len(self._entries)

    def avg_reputation(self) -> float:
        """Return the average reputation score across all calls."""
        if not self._entries:
            return 0.0
        return sum(e.reputation_score for e in self._entries) / len(self._entries)

    def org_costs(self) -> dict[str, float]:
        """Return total trust cost grouped by remote org ID."""
        costs: dict[str, float] = {}
        for e in self._entries:
            costs[e.remote_org_id] = costs.get(e.remote_org_id, 0.0) + e.net_trust_cost
        return costs

    def summary(self) -> dict[str, object]:
        """Return a summary dict of all ledger activity.

        Returns
        -------
        dict
            Keys: ``total_calls``, ``allowed``, ``blocked``, ``block_rate``,
            ``total_trust_cost``, ``avg_surcharge``, ``avg_reputation``.
        """
        return {
            "total_calls": len(self._entries),
            "allowed": self.allowed_count(),
            "blocked": self.blocked_count(),
            "block_rate": round(self.block_rate(), 4),
            "total_trust_cost": round(self.total_trust_cost(), 4),
            "avg_surcharge": round(self.avg_surcharge(), 4),
            "avg_reputation": round(self.avg_reputation(), 4),
        }
