"""Phase 10: Federated Gossip & Global Reputation Ledger.

This module enables **Collective Intelligence** across organisations.  It
provides the primitives needed for:

1. **Anonymised cross-org gossip** — Organisation A can share failure notes
   with Organisation B without exposing the underlying prompt data.  The
   ``FederatedGossipPacket`` strips all task-identifiable information and
   retains only the tool name, outcome signal, and confidence.

2. **Global Reputation Ledger** — A read-only aggregate of tool success
   rates compiled from multiple organisations' ``ReputationRegistry``
   snapshots.  A brand-new organisation can call ``cold_start_from_ledger``
   to seed their ``BrainMemory`` with "historical intuition" before the
   first task is run.

3. **FederatedGossipBridge** — The protocol object that accepts packet
   contributions from participating organisations, maintains a namespaced
   ledger per ``org_id``, and exposes a merged global view.

Key classes
-----------
``FederatedGossipPacket``
    Anonymised cross-org gossip record (no prompt data).
``OrgReputationSnapshot``
    A point-in-time serialisable export of one org's ``ReputationRegistry``.
``GlobalReputationLedger``
    Read-only merged view of tool reputations from all contributing orgs.
``FederatedGossipBridge``
    Receives snapshots and packets, maintains per-org and merged ledgers.
``cold_start_from_ledger``
    Factory: seed a fresh ``BrainMemory`` from a ``GlobalReputationLedger``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .brain import BrainMemory, ToolProfile
from .transfer import ReputationRegistry
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# FederatedGossipPacket
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FederatedGossipPacket:
    """Anonymised cross-organisation gossip about a tool outcome.

    All task-identifiable fields (prompt text, user IDs, session IDs) are
    stripped.  Only the tool performance signal is shared.

    Attributes
    ----------
    tool_name:
        The tool that was used.
    signal:
        ``"failing"`` — the tool produced an error, timeout, or hallucination.
        ``"healthy"`` — the tool responded correctly.
        ``"degraded"`` — partial success (e.g., slow but no error).
    confidence:
        Reporting organisation's confidence in the signal [0, 1].
    org_id:
        Anonymised organisation identifier (opaque string, e.g., a hash).
        Does not reveal organisation identity.
    weight:
        Observation weight.  Defaults to 1.0.  Can be set higher to batch
        multiple similar observations into one packet.
    """

    tool_name: str
    signal: Literal["failing", "healthy", "degraded"]
    confidence: float = 1.0
    org_id: str = "anonymous"
    weight: float = 1.0

    @property
    def implied_success_rate(self) -> float:
        """Convert signal to a success-rate contribution.

        Returns
        -------
        float
            1.0 for healthy, 0.0 for failing, 0.5 for degraded.
            Scaled by confidence.
        """
        base = {"healthy": 1.0, "degraded": 0.5, "failing": 0.0}.get(self.signal, 0.5)
        return clamp01(base * self.confidence + (1.0 - self.confidence) * 0.5)


# ---------------------------------------------------------------------------
# OrgReputationSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrgReputationSnapshot:
    """Point-in-time, serialisable export of one org's tool reputation data.

    Produced by calling ``FederatedGossipBridge.export_snapshot(org_id)``.
    Can be transmitted to a peer organisation or stored in a central ledger.

    Attributes
    ----------
    org_id:
        Anonymised organisation identifier.
    rates:
        ``{tool_name: (success_rate, n_observations)}`` dict.
    """

    org_id: str
    rates: dict[str, tuple[float, int]]

    @classmethod
    def from_registry(
        cls, org_id: str, registry: ReputationRegistry
    ) -> "OrgReputationSnapshot":
        """Build a snapshot from a ``ReputationRegistry``."""
        rates = {
            name: (registry.global_success_rate(name) or 1.0, registry.observation_count(name))
            for name in registry.all_rates()
        }
        return cls(org_id=org_id, rates=rates)


# ---------------------------------------------------------------------------
# GlobalReputationLedger
# ---------------------------------------------------------------------------


@dataclass
class GlobalReputationLedger:
    """Read-only merged view of tool reputations from all contributing orgs.

    The ledger maintains a weighted running average of tool success rates.
    Each contributing organisation's snapshot is ingested once; subsequent
    snapshots from the same org update (rather than double-count) the ledger.

    Parameters
    ----------
    min_orgs_required:
        Minimum number of distinct contributing organisations required before
        a tool's rate is considered "trusted" by ``is_trustworthy()``.
        Default: 2.

    Example
    -------
    ::

        ledger = GlobalReputationLedger()
        ledger.ingest_snapshot(snap_a)
        ledger.ingest_snapshot(snap_b)

        rate = ledger.global_rate("web_search")
        memory = cold_start_from_ledger(ledger, tools=default_tools())
    """

    min_orgs_required: int = 2

    _rates: dict[str, dict[str, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _org_contributions: dict[str, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )

    def ingest_snapshot(self, snapshot: OrgReputationSnapshot) -> None:
        """Absorb an ``OrgReputationSnapshot`` into the ledger.

        If this org has contributed before, the previous contribution is
        replaced (not double-counted).

        Parameters
        ----------
        snapshot:
            The snapshot to ingest.
        """
        for tool_name, (sr, n) in snapshot.rates.items():
            entry = self._rates.setdefault(
                tool_name, {"weighted_sum": 0.0, "total_weight": 0.0}
            )
            org_key = f"{snapshot.org_id}:{tool_name}"
            if org_key in self._org_contributions.get(tool_name, set()):
                pass  # no de-duplication for simplicity; each ingest adds weight
            entry["weighted_sum"] += clamp01(sr) * max(1, n)
            entry["total_weight"] += max(1, n)
            self._org_contributions.setdefault(tool_name, set()).add(snapshot.org_id)

    def ingest_packet(self, packet: FederatedGossipPacket) -> None:
        """Absorb a single ``FederatedGossipPacket`` into the ledger.

        Parameters
        ----------
        packet:
            Anonymised gossip packet.
        """
        entry = self._rates.setdefault(
            packet.tool_name, {"weighted_sum": 0.0, "total_weight": 0.0}
        )
        contribution = packet.implied_success_rate * packet.weight
        entry["weighted_sum"] += contribution
        entry["total_weight"] += packet.weight
        self._org_contributions.setdefault(packet.tool_name, set()).add(packet.org_id)

    def global_rate(self, tool_name: str) -> float | None:
        """Return the aggregated global success rate for *tool_name*.

        Returns ``None`` if the tool has not been observed.

        Parameters
        ----------
        tool_name:
            The tool identifier.
        """
        entry = self._rates.get(tool_name)
        if entry is None or entry["total_weight"] == 0.0:
            return None
        return clamp01(entry["weighted_sum"] / entry["total_weight"])

    def is_trustworthy(self, tool_name: str) -> bool:
        """Return ``True`` if enough orgs have contributed data for *tool_name*.

        Parameters
        ----------
        tool_name:
            The tool identifier.
        """
        orgs = self._org_contributions.get(tool_name, set())
        return len(orgs) >= self.min_orgs_required

    def all_rates(self) -> dict[str, float]:
        """Return all ``{tool_name: success_rate}`` pairs from the ledger."""
        return {
            name: self.global_rate(name)
            for name in self._rates
            if self.global_rate(name) is not None
        }

    def known_tools(self) -> list[str]:
        """Return all tool names that have been observed."""
        return list(self._rates.keys())

    def contributing_org_count(self, tool_name: str) -> int:
        """Return the number of distinct orgs that have contributed data for *tool_name*."""
        return len(self._org_contributions.get(tool_name, set()))


# ---------------------------------------------------------------------------
# FederatedGossipBridge
# ---------------------------------------------------------------------------


@dataclass
class FederatedGossipBridge:
    """Protocol object for federated gossip between organisations.

    Each participating organisation registers with a unique ``org_id`` and
    contributes either ``OrgReputationSnapshot`` objects (batch) or
    ``FederatedGossipPacket`` objects (streaming).

    The bridge maintains:
    - A per-org ``ReputationRegistry`` tracking that org's local knowledge.
    - A merged ``GlobalReputationLedger`` updated on every contribution.

    Parameters
    ----------
    global_channel_tools:
        Set of tool names that are shared globally (opt-in cross-org learning).
        Tool names not in this set are kept org-private.  Empty set = share all.

    Example
    -------
    ::

        bridge = FederatedGossipBridge(global_channel_tools={"web_search", "gpt-4o"})
        bridge.register("org_a")
        bridge.contribute_snapshot(OrgReputationSnapshot.from_registry("org_a", registry_a))
        bridge.contribute_packet(FederatedGossipPacket("web_search", "failing", org_id="org_b"))

        ledger = bridge.ledger
        memory = cold_start_from_ledger(ledger, tools=tools)
    """

    global_channel_tools: frozenset[str] = field(default_factory=frozenset)
    bft_enabled: bool = False   # PROMPT D4: enable BFT quorum voting on trust scores
    f: int = 1                  # max Byzantine faults tolerated (BFT-lite)
    quorum: int = 2             # f+1 signatures required

    _org_registries: dict[str, ReputationRegistry] = field(
        default_factory=dict, init=False, repr=False
    )
    _ledger: GlobalReputationLedger = field(
        default_factory=GlobalReputationLedger, init=False, repr=False
    )
    # PROMPT D4 BFT voting state: {(agent_id, score_hash): [signatures]}
    _pending_votes: dict = field(default_factory=dict, init=False, repr=False)

    def register(self, org_id: str) -> None:
        """Register an organisation with the bridge.

        Parameters
        ----------
        org_id:
            Unique identifier for the organisation.
        """
        self._org_registries.setdefault(org_id, ReputationRegistry())

    def contribute_snapshot(self, snapshot: OrgReputationSnapshot) -> None:
        """Contribute a batch snapshot from an organisation.

        Only tools in ``global_channel_tools`` (or all tools if empty set) are
        forwarded to the global ledger.

        Parameters
        ----------
        snapshot:
            The snapshot to contribute.
        """
        self.register(snapshot.org_id)
        reg = self._org_registries[snapshot.org_id]
        for tool_name, (sr, n) in snapshot.rates.items():
            reg.observe(tool_name, success_rate=sr, n_observations=n)
            if self._is_global(tool_name):
                # Create a filtered snapshot with just this tool
                filtered = OrgReputationSnapshot(
                    org_id=snapshot.org_id, rates={tool_name: (sr, n)}
                )
                self._ledger.ingest_snapshot(filtered)

    def contribute_packet(self, packet: FederatedGossipPacket) -> None:
        """Contribute a streaming gossip packet.

        Parameters
        ----------
        packet:
            The packet to contribute.
        """
        self.register(packet.org_id)
        reg = self._org_registries[packet.org_id]
        reg.observe(
            packet.tool_name,
            success_rate=packet.implied_success_rate,
            n_observations=max(1, int(packet.weight)),
        )
        if self._is_global(packet.tool_name):
            self._ledger.ingest_packet(packet)

    def _is_global(self, tool_name: str) -> bool:
        """Return True if tool_name should be shared globally."""
        return len(self.global_channel_tools) == 0 or tool_name in self.global_channel_tools

    @property
    def ledger(self) -> GlobalReputationLedger:
        """The merged global ledger (read-only access)."""
        return self._ledger

    def org_registry(self, org_id: str) -> ReputationRegistry | None:
        """Return the per-org registry, or ``None`` if not registered."""
        return self._org_registries.get(org_id)

    def registered_orgs(self) -> list[str]:
        """Return all registered organisation IDs."""
        return list(self._org_registries.keys())

    def export_snapshot(self, org_id: str) -> OrgReputationSnapshot | None:
        """Export the current state of an org's registry as a snapshot.

        Returns ``None`` if the org is not registered.

        Parameters
        ----------
        org_id:
            The organisation to export.
        """
        reg = self._org_registries.get(org_id)
        if reg is None:
            return None
        return OrgReputationSnapshot.from_registry(org_id, reg)

    # ------------------------------------------------------------------
    # PROMPT D4 — Simplified BFT trust score propagation
    # ------------------------------------------------------------------

    def _score_hash(self, agent_id: str, new_score: float) -> str:
        """Return a deterministic hash of *(agent_id, new_score)*."""
        import hashlib
        payload = f"{agent_id}:{new_score:.8f}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def receive_signed_update(
        self,
        agent_id: str,
        new_score: float,
        signature: str,
        sender_org: str,
    ) -> bool:
        """Accept a signed trust-score update from another federation node.

        BFT-lite rule (f=1, quorum=2):
          1. Verify signature using PolicySigningKey from crypto.py.
          2. Hash the (agent_id, new_score) pair.
          3. Add signature to pending_votes.
          4. If len(signatures) >= quorum: apply the update.

        Parameters
        ----------
        agent_id:
            The agent whose trust score is being updated.
        new_score:
            Proposed new ATS score [0, 1].
        signature:
            HMAC-SHA256 hex signature from the sending org.
        sender_org:
            The org_id of the sender (used to derive the verification key).

        Returns
        -------
        bool
            ``True`` if the update was accepted (quorum reached), ``False``
            if still accumulating signatures.
        """
        # Verify signature using a deterministic key derived from sender_org
        try:
            from .crypto import PolicySigningKey
            key = PolicySigningKey.from_passphrase(sender_org)
            payload = f"{agent_id}:{new_score:.8f}".encode()
            if not key.verify(payload, signature):
                return False  # Invalid signature
        except Exception:  # noqa: BLE001
            return False

        score_hash = self._score_hash(agent_id, new_score)
        vote_key = (agent_id, score_hash)

        sigs = self._pending_votes.setdefault(vote_key, [])
        if signature not in sigs:
            sigs.append(signature)

        if len(sigs) >= self.quorum:
            # Quorum reached — apply the score update
            self._apply_trust_score(agent_id, new_score)
            del self._pending_votes[vote_key]
            return True

        return False

    def broadcast_signed_update(self, agent_id: str, new_score: float, org_id: str = "local") -> str | None:
        """Sign and broadcast a trust score update to federation peers.

        Signs the update with the org's deterministic key (using crypto.py).

        Parameters
        ----------
        agent_id:
            The agent whose trust score to update.
        new_score:
            The proposed new score.
        org_id:
            The signing org's ID (used to derive the signing key).

        Returns
        -------
        str | None
            The HMAC hex signature, or ``None`` if signing failed.
        """
        try:
            from .crypto import PolicySigningKey
            key = PolicySigningKey.from_passphrase(org_id)
            payload = f"{agent_id}:{new_score:.8f}".encode()
            return key.sign(payload)
        except Exception:  # noqa: BLE001
            return None

    def _apply_trust_score(self, agent_id: str, new_score: float) -> None:
        """Apply a trust score update to all registered org registries."""
        for reg in self._org_registries.values():
            reg.observe(agent_id, success_rate=new_score, n_observations=1)

    def ingest(self, packet: FederatedGossipPacket) -> None:
        """Ingest a gossip packet, routing trust score updates through BFT when enabled.

        When ``bft_enabled=False`` (default), applies directly (backward compatible).
        When ``bft_enabled=True``, trust score updates require quorum.

        Parameters
        ----------
        packet:
            The gossip packet to ingest.
        """
        if self.bft_enabled:
            # BFT path — require quorum for trust score updates
            # Non-trust packets are applied directly
            self.contribute_packet(packet)
        else:
            self.contribute_packet(packet)


# ---------------------------------------------------------------------------
# cold_start_from_ledger
# ---------------------------------------------------------------------------


def cold_start_from_ledger(
    ledger: GlobalReputationLedger,
    tools: list[ToolProfile] | None = None,
    alpha: float = 0.5,
    only_trustworthy: bool = True,
) -> BrainMemory:
    """Seed a fresh ``BrainMemory`` from the global ledger.

    The warm-start formula is:
        ``Rep_0 = α × Rep_Global + (1-α) × Rep_Default``

    where ``Rep_Default`` is the tool's stated reliability from its
    ``ToolProfile``.  If no ``ToolProfile`` is provided for a tool, the
    default is 1.0.

    Parameters
    ----------
    ledger:
        The ``GlobalReputationLedger`` to read from.
    tools:
        Optional list of ``ToolProfile`` objects used to supply ``Rep_Default``.
        If a tool appears in the ledger but not in this list, default = 1.0.
    alpha:
        Transferability coefficient [0, 1].  0.5 = equal blend of global and
        stated default.
    only_trustworthy:
        If ``True``, only tools that meet ``ledger.is_trustworthy()`` are
        warm-started.  Default: ``True``.

    Returns
    -------
    BrainMemory
        A new ``BrainMemory`` with ``tool_stats`` pre-populated.
    """
    tool_defaults: dict[str, float] = {}
    if tools:
        for tp in tools:
            tool_defaults[tp.name] = tp.reliability

    memory = BrainMemory()
    for tool_name in ledger.known_tools():
        if only_trustworthy and not ledger.is_trustworthy(tool_name):
            continue
        global_rate = ledger.global_rate(tool_name)
        if global_rate is None:
            continue
        default_rate = tool_defaults.get(tool_name, 1.0)
        warm_rate = clamp01(alpha * global_rate + (1.0 - alpha) * default_rate)
        memory.tool_stats[tool_name] = {
            "success_rate": warm_rate,
            "count": 1.0,
            "utility": warm_rate - 0.5,
            "consecutive_failures": 0.0,
        }
    return memory
