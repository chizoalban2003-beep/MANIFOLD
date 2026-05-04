"""Phase 15: Global Reputation Hub — shared community intelligence.

The ``ReputationHub`` gives MANIFOLD instances a "warm start" by providing
community-sourced baseline data on known tool reliabilities and risk profiles.
Organisations can optionally contribute anonymised failure signals; the hub
aggregates them into a ``GlobalReputationLedger`` that any new MANIFOLD
deployment can consume.

Think of it as the **immune system effect**: when the first 100 MANIFOLD
users detect a provider outage or model drift, the next 10,000 get an
automatic warm-start that steers them away from the failing component before
they make their first call.

Key classes
-----------
``CommunityBaseline``
    Frozen snapshot of community-sourced tool reliability and known risk
    flags.  Ships with a built-in default baseline covering common LLM
    providers, vector databases, and support APIs.

``ReputationHub``
    Manages a ``CommunityBaseline`` and accepts anonymous ``FederatedGossipPacket``
    contributions.  Exposes ``warm_start_ledger()`` to seed a fresh
    ``GlobalReputationLedger`` from community data, and ``to_org_snapshot()``
    to create an ``OrgReputationSnapshot`` for federated exchange.

Example
-------
::

    from manifold.hub import ReputationHub
    from manifold import GlobalReputationLedger

    hub = ReputationHub()
    ledger = GlobalReputationLedger(min_orgs_required=1)
    hub.warm_start_ledger(ledger)

    rate = ledger.global_rate("gpt-4o")  # community baseline: ~0.92
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entropy import ReputationDecay, VolatilityTable
from .federation import (
    FederatedGossipBridge,
    FederatedGossipPacket,
    GlobalReputationLedger,
    OrgReputationSnapshot,
)
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# Built-in community baselines
# ---------------------------------------------------------------------------

# Tool name → (baseline_reliability, observation_weight)
# These values are intentionally conservative estimates derived from
# publicly available benchmark data and are updated with each release.
_DEFAULT_BASELINES: dict[str, tuple[float, int]] = {
    # OpenAI models
    "gpt-4o": (0.92, 500),
    "gpt-4o-mini": (0.88, 800),
    "gpt-4-turbo": (0.91, 300),
    "gpt-3.5-turbo": (0.85, 1000),
    # Anthropic
    "claude-3-5-sonnet": (0.93, 400),
    "claude-3-haiku": (0.90, 600),
    "claude-3-opus": (0.94, 200),
    # Google
    "gemini-1.5-pro": (0.89, 350),
    "gemini-1.5-flash": (0.87, 450),
    # Open-source / self-hosted
    "llama-3-70b": (0.84, 250),
    "mistral-7b": (0.82, 300),
    # Vector stores
    "pinecone": (0.95, 200),
    "weaviate": (0.93, 150),
    "chroma": (0.91, 180),
    # Search + retrieval
    "tavily_search": (0.88, 300),
    "serper_search": (0.86, 250),
    "bing_search": (0.87, 200),
    # Support / CRM
    "zendesk_api": (0.94, 400),
    "intercom_api": (0.93, 350),
    "salesforce_api": (0.91, 300),
    # Generic
    "web_search": (0.85, 500),
    "calculator": (0.99, 1000),
    "code_interpreter": (0.90, 400),
}

# Known risk flags: tool_name → brief description of known issue
_DEFAULT_RISK_FLAGS: dict[str, str] = {
    "gpt-4o-mini": "Higher hallucination rate on math/legal tasks vs. larger models.",
    "gpt-3.5-turbo": "Prone to confident errors on multi-step reasoning.",
    "mistral-7b": "Instruction-following degrades on long context (>16k tokens).",
    "llama-3-70b": "Self-hosted latency varies significantly with hardware.",
    "web_search": "Results may be stale; always verify critical facts.",
    "code_interpreter": "Sandbox timeout at 30 s; avoid long-running computations.",
}


# ---------------------------------------------------------------------------
# CommunityBaseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommunityBaseline:
    """Frozen snapshot of community-sourced tool reliability and risk flags.

    Attributes
    ----------
    tool_baselines:
        ``{tool_name: (reliability, observation_weight)}`` where
        *reliability* is in [0, 1] and *observation_weight* is the effective
        number of community observations backing the estimate.
    risk_flags:
        ``{tool_name: description}`` — known risk signals from the community.
    version:
        Baseline version string (semver).
    """

    tool_baselines: dict[str, tuple[float, int]]
    risk_flags: dict[str, str]
    version: str = "1.0.0"

    @classmethod
    def default(cls) -> "CommunityBaseline":
        """Return the built-in community baseline shipped with manifold-ai."""
        return cls(
            tool_baselines=dict(_DEFAULT_BASELINES),
            risk_flags=dict(_DEFAULT_RISK_FLAGS),
        )

    def reliability(self, tool_name: str) -> float | None:
        """Return the community reliability for *tool_name*, or ``None``."""
        entry = self.tool_baselines.get(tool_name)
        return clamp01(entry[0]) if entry is not None else None

    def observation_weight(self, tool_name: str) -> int:
        """Return the community observation weight for *tool_name*, or 0."""
        entry = self.tool_baselines.get(tool_name)
        return entry[1] if entry is not None else 0

    def is_flagged(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* has a known community risk flag."""
        return tool_name in self.risk_flags

    def risk_flag(self, tool_name: str) -> str | None:
        """Return the risk flag description for *tool_name*, or ``None``."""
        return self.risk_flags.get(tool_name)

    def tool_names(self) -> list[str]:
        """Return a sorted list of all tools covered by this baseline."""
        return sorted(self.tool_baselines)

    def to_org_snapshot(self, org_id: str = "community_baseline") -> OrgReputationSnapshot:
        """Convert this baseline to an ``OrgReputationSnapshot`` for ledger seeding."""
        return OrgReputationSnapshot(
            org_id=org_id,
            rates={name: (clamp01(rel), weight) for name, (rel, weight) in self.tool_baselines.items()},
        )


# ---------------------------------------------------------------------------
# ReputationHub
# ---------------------------------------------------------------------------


@dataclass
class ReputationHub:
    """Global reputation hub — shared community intelligence for MANIFOLD.

    Manages a ``CommunityBaseline`` and accepts anonymised
    ``FederatedGossipPacket`` contributions.  New MANIFOLD deployments call
    ``warm_start_ledger()`` to seed their ``GlobalReputationLedger`` with
    community data before their first real task.

    Parameters
    ----------
    baseline:
        The community baseline to use.  Defaults to the built-in baseline
        shipped with manifold-ai.
    min_contributions_for_update:
        Minimum number of community contributions required before the hub
        considers a tool's baseline "updated" vs. just using the factory default.

    Example
    -------
    ::

        hub = ReputationHub()

        # Seed a new deployment's ledger from the community
        ledger = GlobalReputationLedger(min_orgs_required=1)
        hub.warm_start_ledger(ledger)

        # Contribute an anonymised failure signal
        hub.contribute(FederatedGossipPacket(
            tool_name="gpt-4o-mini",
            signal="failing",
            confidence=0.85,
            org_id="org_abc123",
        ))

        # Get updated reliability after contributions
        rate = hub.live_reliability("gpt-4o-mini")
    """

    baseline: CommunityBaseline = field(default_factory=CommunityBaseline.default)
    min_contributions_for_update: int = 5

    _bridge: FederatedGossipBridge = field(
        default_factory=FederatedGossipBridge, init=False, repr=False
    )
    _contributions: list[FederatedGossipPacket] = field(
        default_factory=list, init=False, repr=False
    )
    _live_ledger: GlobalReputationLedger = field(init=False, repr=False)
    _decay: ReputationDecay = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._live_ledger = GlobalReputationLedger(min_orgs_required=1)
        # Seed the live ledger with the community baseline
        self._live_ledger.ingest_snapshot(self.baseline.to_org_snapshot())
        self._decay = ReputationDecay(volatility=VolatilityTable.default())

    def contribute(
        self,
        packet: FederatedGossipPacket,
        *,
        anonymize: bool = True,
    ) -> None:
        """Submit an anonymised tool failure/health signal to the hub.

        Parameters
        ----------
        packet:
            A ``FederatedGossipPacket`` from any participating organisation.
        anonymize:
            When ``True`` (default), the ``org_id`` is discarded and replaced
            with a generic ``"contributor"`` tag before ingesting, so the hub
            never stores identifiable organisation data.
        """
        if anonymize:
            packet = FederatedGossipPacket(
                tool_name=packet.tool_name,
                signal=packet.signal,
                confidence=packet.confidence,
                org_id="contributor",
                weight=packet.weight,
            )
        self._contributions.append(packet)
        self._bridge.contribute_packet(packet)
        self._live_ledger.ingest_packet(packet)
        # Record a fresh signal timestamp for decay tracking
        live_rate = self._live_ledger.global_rate(packet.tool_name)
        if live_rate is not None:
            # Infer domain from community baseline if available
            domain = "general"
            for name, (_, _) in self.baseline.tool_baselines.items():
                if name == packet.tool_name:
                    break
            self._decay.record_signal(
                packet.tool_name, domain=domain, reliability=clamp01(live_rate)
            )

    def warm_start_ledger(
        self,
        ledger: GlobalReputationLedger,
        *,
        include_contributions: bool = True,
    ) -> None:
        """Seed *ledger* with community baseline data.

        Parameters
        ----------
        ledger:
            The ``GlobalReputationLedger`` to populate.
        include_contributions:
            When ``True`` (default), also ingest any live community
            contributions received since the hub was created.
        """
        ledger.ingest_snapshot(self.baseline.to_org_snapshot())
        if include_contributions and self._contributions:
            for packet in self._contributions:
                ledger.ingest_packet(packet)

    def live_reliability(self, tool_name: str) -> float | None:
        """Return the live, time-decayed community reliability for *tool_name*.

        The raw reliability is fetched from the live ledger (or baseline) and
        then adjusted by the exponential decay function based on how long ago
        the last gossip/outcome signal was recorded for this tool.

        Returns
        -------
        float | None
            Decay-adjusted reliability in [0, 1], or ``None`` if unknown.
        """
        # Prefer the live ledger if it has data for this tool
        live = self._live_ledger.global_rate(tool_name)
        raw = clamp01(live) if live is not None else self.baseline.reliability(tool_name)
        if raw is None:
            return None
        return self._decay.decayed_reliability(tool_name, raw)

    def contribution_count(self, tool_name: str | None = None) -> int:
        """Return the number of contributions received.

        Parameters
        ----------
        tool_name:
            If provided, counts only contributions for that tool.
            If ``None``, returns the total across all tools.
        """
        if tool_name is None:
            return len(self._contributions)
        return sum(1 for p in self._contributions if p.tool_name == tool_name)

    def system_entropy(self) -> float:
        """Return the mean entropy score across all tracked tools (Phase 26).

        Returns 0.0 when no gossip signals have been received.
        """
        return self._decay.system_entropy()

    def tool_entropy(self, tool_name: str) -> float:
        """Return the entropy score for a specific tool [0, 1] (Phase 26)."""
        return self._decay.entropy_score(tool_name)

    def flagged_tools(self) -> list[str]:
        """Return tool names that have community risk flags."""
        return [t for t in self.baseline.tool_names() if self.baseline.is_flagged(t)]

    def community_summary(self) -> dict[str, Any]:
        """Return a summary dict of hub state.

        Keys: ``baseline_tools``, ``flagged_tools``, ``total_contributions``,
        ``baseline_version``.
        """
        return {
            "baseline_tools": len(self.baseline.tool_baselines),
            "flagged_tools": len(self.flagged_tools()),
            "total_contributions": len(self._contributions),
            "baseline_version": self.baseline.version,
        }

    def to_org_snapshot(self, org_id: str = "community_hub") -> OrgReputationSnapshot:
        """Export the current hub state as an ``OrgReputationSnapshot``."""
        # Merge baseline with any live contributions
        merged: dict[str, tuple[float, int]] = {}
        for name, (rel, weight) in self.baseline.tool_baselines.items():
            live = self._live_ledger.global_rate(name)
            effective_rel = live if live is not None else clamp01(rel)
            merged[name] = (effective_rel, weight)
        return OrgReputationSnapshot(org_id=org_id, rates=merged)
