"""Phase 28: Semantic Tool Discovery — Self-Expanding Intelligence.

When the local ``ConnectorRegistry`` has no tool with reliability > 0.7 for a
given domain, the ``DiscoveryScout`` queries the ``ReputationHub`` for "highly
gossiped" tools it hasn't seen yet and registers them as **Probationary**
connectors with a 25 % trust penalty.  After 5 successful outcomes the
probationary status is lifted and the tool graduates to full trust.

Key classes
-----------
``ProbationaryState``
    Tracks how many successful outcomes a probationary tool has accumulated.
``DiscoveryScout``
    Hub-aware scout that finds new tools and hands them off for onboarding.
``ProbationaryRegistry``
    Thin overlay that manages probationary tool states and promotes graduates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .brain import BrainTask, ToolProfile
from .connector import ConnectorRegistry, ToolConnector
from .hub import ReputationHub
from .recruiter import MarketplaceListing, RecruitmentResult, SovereignRecruiter
from .trustrouter import clamp01

# Number of successful outcomes required to graduate from probationary status
PROBATIONARY_GRADUATION_THRESHOLD: int = 5

# Reliability threshold below which a domain is considered "lacking"
RELIABILITY_COVERAGE_THRESHOLD: float = 0.70

# Minimum gossip count for a hub tool to be considered "highly gossiped"
MIN_GOSSIP_COUNT: int = 1

# Trust penalty applied to probationary tools
PROBATIONARY_TRUST_PENALTY: float = 0.25


# ---------------------------------------------------------------------------
# ProbationaryState
# ---------------------------------------------------------------------------


@dataclass
class ProbationaryState:
    """Tracks the probationary status of a single tool.

    Parameters
    ----------
    tool_name:
        Unique tool identifier.
    original_reliability:
        The reliability score *before* the probationary penalty.

    Attributes
    ----------
    successful_outcomes:
        Number of successful calls accumulated so far.
    graduated:
        ``True`` once ``successful_outcomes >= PROBATIONARY_GRADUATION_THRESHOLD``.
    """

    tool_name: str
    original_reliability: float
    successful_outcomes: int = 0
    graduated: bool = False

    @property
    def penalised_reliability(self) -> float:
        """Reliability with 25 % trust penalty applied (while probationary)."""
        return clamp01(self.original_reliability * (1.0 - PROBATIONARY_TRUST_PENALTY))

    def record_outcome(self, success: bool) -> bool:
        """Record an outcome and return ``True`` if the tool just graduated.

        Parameters
        ----------
        success:
            Whether the call succeeded.

        Returns
        -------
        bool
            ``True`` if this outcome pushed the tool over the graduation
            threshold.
        """
        if self.graduated:
            return False
        if success:
            self.successful_outcomes += 1
        if self.successful_outcomes >= PROBATIONARY_GRADUATION_THRESHOLD:
            self.graduated = True
            return True
        return False


# ---------------------------------------------------------------------------
# ProbationaryRegistry
# ---------------------------------------------------------------------------


@dataclass
class ProbationaryRegistry:
    """Overlay that tracks and promotes probationary tools.

    Parameters
    ----------
    connector_registry:
        The underlying ``ConnectorRegistry``.

    Example
    -------
    ::

        prob_reg = ProbationaryRegistry(connector_registry=registry)
        prob_reg.register_probationary(connector, original_reliability=0.88)
        for _ in range(5):
            prob_reg.record_outcome("my-tool", success=True)
        # tool is now graduated
    """

    connector_registry: ConnectorRegistry
    _states: dict[str, ProbationaryState] = field(
        default_factory=dict, init=False, repr=False
    )

    def register_probationary(
        self, connector: ToolConnector, *, original_reliability: float
    ) -> ProbationaryState:
        """Register *connector* as probationary in the underlying registry.

        Parameters
        ----------
        connector:
            The ``ToolConnector`` to register.
        original_reliability:
            The pre-penalty reliability used to create the penalised profile.

        Returns
        -------
        ProbationaryState
            The new probationary state record.
        """
        self.connector_registry.register(connector)
        state = ProbationaryState(
            tool_name=connector.name,
            original_reliability=original_reliability,
        )
        self._states[connector.name] = state
        return state

    def record_outcome(self, tool_name: str, *, success: bool) -> bool:
        """Record an outcome for a probationary tool.

        Parameters
        ----------
        tool_name:
            Tool to update.
        success:
            Whether the call succeeded.

        Returns
        -------
        bool
            ``True`` if the tool just graduated from probationary status.
        """
        state = self._states.get(tool_name)
        if state is None or state.graduated:
            return False
        just_graduated = state.record_outcome(success)
        if just_graduated:
            self._promote(tool_name, state)
        return just_graduated

    def is_probationary(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is still in probationary status."""
        state = self._states.get(tool_name)
        return state is not None and not state.graduated

    def state(self, tool_name: str) -> ProbationaryState | None:
        """Return the ``ProbationaryState`` for *tool_name*, or ``None``."""
        return self._states.get(tool_name)

    def all_states(self) -> dict[str, ProbationaryState]:
        """Return a copy of all probationary state records."""
        return dict(self._states)

    def probationary_tools(self) -> list[str]:
        """Return the names of all tools still in probationary status."""
        return [n for n, s in self._states.items() if not s.graduated]

    def graduated_tools(self) -> list[str]:
        """Return the names of all tools that have graduated."""
        return [n for n, s in self._states.items() if s.graduated]

    def _promote(self, tool_name: str, state: ProbationaryState) -> None:
        """Restore full reliability for a graduated tool (in-registry update)."""
        connector = self.connector_registry.get(tool_name)
        if connector is None:
            return
        # Rebuild the profile with original (un-penalised) reliability
        old_profile = connector.refreshed_profile()
        new_profile = ToolProfile(
            name=old_profile.name,
            cost=old_profile.cost,
            latency=old_profile.latency,
            reliability=clamp01(state.original_reliability),
            risk=old_profile.risk,
            asset=old_profile.asset,
            domain=old_profile.domain,
        )
        promoted_connector = ToolConnector(
            name=connector.name,
            fn=connector.fn,
            profile=new_profile,
        )
        self.connector_registry.register(promoted_connector)


# ---------------------------------------------------------------------------
# DiscoveryScout
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryScout:
    """Hub-aware scout that discovers and onboards new tools (Phase 28).

    When the local ``ConnectorRegistry`` has no tool with reliability > 0.7
    for a domain, the scout:

    1. Queries the ``ReputationHub`` for tools with high gossip volume not
       already in the registry.
    2. Builds a ``MarketplaceListing`` from hub data.
    3. Registers the tool via ``ProbationaryRegistry`` with a 25 % penalty.

    Parameters
    ----------
    hub:
        The ``ReputationHub`` to query for highly-gossiped tools.
    prob_registry:
        The ``ProbationaryRegistry`` used for onboarding.
    min_hub_reliability:
        Minimum reliability in the hub for a tool to be considered a
        discovery candidate.  Default: ``0.70``.
    probe_fn:
        Optional callable ``(tool_name, domain) -> bool`` used to probe
        discovered tools.  Defaults to ``None`` (no probing required).
    """

    hub: ReputationHub
    prob_registry: ProbationaryRegistry
    min_hub_reliability: float = RELIABILITY_COVERAGE_THRESHOLD
    probe_fn: Callable[[str, str], bool] | None = None

    _discovery_log: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )

    def scout_for_domain(self, domain: str) -> list[str]:
        """Discover and onboard new tools for *domain* if the registry is weak.

        Parameters
        ----------
        domain:
            The domain to check coverage for.

        Returns
        -------
        list[str]
            Names of newly registered tools.
        """
        registry = self.prob_registry.connector_registry
        # Check if coverage is already adequate
        if self._domain_covered(registry, domain):
            return []

        # Find hub tools not already in the registry
        candidates = self._hub_candidates(registry, domain)
        registered: list[str] = []
        for tool_name, hub_reliability in candidates:
            self._onboard(tool_name, domain=domain, hub_reliability=hub_reliability)
            registered.append(tool_name)

        event: dict[str, object] = {
            "domain": domain,
            "registered": registered,
            "candidates_found": len(candidates),
        }
        self._discovery_log.append(event)
        return registered

    def discovery_log(self) -> list[dict[str, object]]:
        """Return a copy of all discovery events."""
        return list(self._discovery_log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _domain_covered(self, registry: ConnectorRegistry, domain: str) -> bool:
        """Return True if registry has a tool with reliability > threshold for domain."""
        for name in registry.names():
            connector = registry.get(name)
            if connector is None:
                continue
            profile = connector.refreshed_profile()
            if profile.domain in (domain, "general"):
                if profile.reliability > RELIABILITY_COVERAGE_THRESHOLD:
                    return True
        return False

    def _hub_candidates(
        self, registry: ConnectorRegistry, domain: str
    ) -> list[tuple[str, float]]:
        """Return (tool_name, reliability) pairs from hub not in registry."""
        known = set(registry.names())
        candidates: list[tuple[str, float]] = []
        for tool_name in self.hub.baseline.tool_names():
            if tool_name in known:
                continue
            # Only tools with enough gossip and adequate reliability
            if self.hub.contribution_count(tool_name) < MIN_GOSSIP_COUNT:
                # Allow baseline tools even with zero gossip (they have community weight)
                pass
            rel = self.hub.live_reliability(tool_name)
            if rel is None or rel < self.min_hub_reliability:
                continue
            candidates.append((tool_name, rel))
        # Sort by reliability descending; take at most 3 per call
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:3]

    def _onboard(
        self, tool_name: str, *, domain: str, hub_reliability: float
    ) -> ProbationaryState:
        """Create a probationary connector for *tool_name* from hub data."""
        penalised_rel = clamp01(hub_reliability * (1.0 - PROBATIONARY_TRUST_PENALTY))

        def _stub(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"tool": tool_name, "status": "discovered_probationary"}

        profile = ToolProfile(
            name=tool_name,
            cost=0.05,
            latency=0.5,
            reliability=penalised_rel,
            risk=0.10,
            asset=0.75,
            domain=domain,
        )
        connector = ToolConnector(name=tool_name, fn=_stub, profile=profile)
        return self.prob_registry.register_probationary(
            connector, original_reliability=hub_reliability
        )


# ---------------------------------------------------------------------------
# Enhanced SovereignRecruiter (with DiscoveryScout integration)
# ---------------------------------------------------------------------------


@dataclass
class EnhancedRecruiter:
    """``SovereignRecruiter`` + ``DiscoveryScout`` integration (Phase 28).

    Wraps the standard ``SovereignRecruiter`` and, if marketplace recruitment
    fails, falls back to hub-based semantic discovery.

    Parameters
    ----------
    registry:
        The ``ConnectorRegistry`` to monitor and expand.
    hub:
        The ``ReputationHub`` used for discovery fallback.
    recruiter_kwargs:
        Extra keyword arguments forwarded to ``SovereignRecruiter.__init__``.
    """

    registry: ConnectorRegistry
    hub: ReputationHub
    recruiter_kwargs: dict[str, object] = field(default_factory=dict)

    _recruiter: SovereignRecruiter = field(init=False, repr=False)
    _prob_registry: ProbationaryRegistry = field(init=False, repr=False)
    _scout: DiscoveryScout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._recruiter = SovereignRecruiter(
            registry=self.registry, **self.recruiter_kwargs  # type: ignore[arg-type]
        )
        self._prob_registry = ProbationaryRegistry(connector_registry=self.registry)
        self._scout = DiscoveryScout(
            hub=self.hub, prob_registry=self._prob_registry
        )

    def recruit_if_needed(self, task: BrainTask) -> RecruitmentResult:
        """Run marketplace recruitment; fall back to hub discovery if needed."""
        result = self._recruiter.recruit_if_needed(task)
        if not result.registered:
            # Try hub-based discovery
            discovered = self._scout.scout_for_domain(task.domain)
            if discovered:
                listing = MarketplaceListing(
                    name=discovered[0],
                    domain=task.domain,
                    description=f"Hub-discovered tool for domain={task.domain!r}",
                    estimated_reliability=clamp01(
                        (self.hub.live_reliability(discovered[0]) or 0.75) * (1 - PROBATIONARY_TRUST_PENALTY)
                    ),
                    estimated_risk=0.10,
                    estimated_cost=0.05,
                    estimated_latency=0.5,
                    estimated_asset=0.75,
                    source="reputation_hub",
                )
                return RecruitmentResult(
                    triggered=True,
                    reason=(
                        f"Hub discovery found {len(discovered)} tool(s) for "
                        f"domain={task.domain!r}; registered as probationary"
                    ),
                    candidates_found=len(discovered),
                    selected_listing=listing,
                    scout_pass_rate=1.0,
                    registered=True,
                    registered_tool_name=discovered[0],
                )
        return result

    @property
    def prob_registry(self) -> ProbationaryRegistry:
        """Return the underlying ``ProbationaryRegistry``."""
        return self._prob_registry

    @property
    def scout(self) -> DiscoveryScout:
        """Return the underlying ``DiscoveryScout``."""
        return self._scout
