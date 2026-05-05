"""Phase 17: Sovereign Recruiter — MANIFOLD's self-expanding capability layer.

When every tool in the registry is under-performing for a high-complexity task,
the ``SovereignRecruiter`` doesn't just refuse — it **searches for new talent**,
profiles the candidates, and on-boards the best one as a probationary asset.

Decision rule
-------------
.. math::

    \\text{Recruit} \\iff Complexity > \\theta_c
    \\;\\land\\; \\forall t \\in Registry: Reliability(t) < \\theta_r

Discovery pipeline
------------------
1. **Marketplace Search** — query a simulated (or real) tool catalog for
   candidates matching the task domain.
2. **Zero-Shot Profiling** — evaluate candidate metadata to produce a
   ``ToolProfile`` with a conservative (probationary) reliability score.
3. **Scout Tasks** — run *n* internal probe calls to establish a baseline
   before exposing the new tool to real traffic.
4. **Registration** — register the best-scoring candidate as a
   ``ToolConnector`` tagged ``"probationary"``.

Key classes
-----------
``MarketplaceListing``
    Metadata for a candidate tool discovered during marketplace search.
``RecruitmentResult``
    Outcome of a single recruitment run: what was found, scouted, registered.
``SovereignRecruiter``
    Main recruiter that monitors the registry and triggers discovery when
    needed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .brain import BrainTask, ToolProfile
from .connector import ConnectorRegistry, ToolConnector
from .trustrouter import clamp01


# ---------------------------------------------------------------------------
# Default tool marketplace catalog
# ---------------------------------------------------------------------------

# A lightweight built-in catalog of tools that can be "discovered".
# Each entry: name → {domain, reliability_estimate, risk, cost, latency, asset}
_DEFAULT_MARKETPLACE: dict[str, dict[str, object]] = {
    # LLM providers
    "cohere-command": {
        "domain": "language",
        "reliability": 0.85,
        "risk": 0.10,
        "cost": 0.04,
        "latency": 0.8,
        "asset": 0.75,
        "description": "Cohere Command — reliable language generation",
    },
    "together-llama3": {
        "domain": "language",
        "reliability": 0.82,
        "risk": 0.12,
        "cost": 0.02,
        "latency": 1.0,
        "asset": 0.72,
        "description": "Llama-3 via Together AI — open-source backbone",
    },
    "mistral-large": {
        "domain": "language",
        "reliability": 0.88,
        "risk": 0.09,
        "cost": 0.06,
        "latency": 0.9,
        "asset": 0.80,
        "description": "Mistral Large — European open-weight flagship",
    },
    # Specialised tools
    "wolfram-alpha": {
        "domain": "math",
        "reliability": 0.97,
        "risk": 0.02,
        "cost": 0.01,
        "latency": 0.5,
        "asset": 0.90,
        "description": "Wolfram Alpha — computational knowledge engine",
    },
    "code-interpreter-v2": {
        "domain": "code",
        "reliability": 0.91,
        "risk": 0.08,
        "cost": 0.05,
        "latency": 1.2,
        "asset": 0.85,
        "description": "Code Interpreter v2 — sandboxed Python execution",
    },
    "serp-api-pro": {
        "domain": "search",
        "reliability": 0.94,
        "risk": 0.05,
        "cost": 0.02,
        "latency": 0.4,
        "asset": 0.78,
        "description": "SERP API Pro — real-time search index",
    },
    "legal-search-pro": {
        "domain": "legal",
        "reliability": 0.90,
        "risk": 0.06,
        "cost": 0.08,
        "latency": 1.0,
        "asset": 0.88,
        "description": "Legal Search Pro — case law and statute retrieval",
    },
    "finance-data-api": {
        "domain": "finance",
        "reliability": 0.93,
        "risk": 0.07,
        "cost": 0.06,
        "latency": 0.3,
        "asset": 0.85,
        "description": "Finance Data API — real-time market and accounting data",
    },
    "medical-kb": {
        "domain": "medical",
        "reliability": 0.89,
        "risk": 0.11,
        "cost": 0.10,
        "latency": 0.9,
        "asset": 0.92,
        "description": "Medical Knowledge Base — clinical guideline retrieval",
    },
    "translation-api": {
        "domain": "translation",
        "reliability": 0.95,
        "risk": 0.03,
        "cost": 0.02,
        "latency": 0.3,
        "asset": 0.80,
        "description": "Translation API — 100+ language neural translation",
    },
    "vector-store-v3": {
        "domain": "retrieval",
        "reliability": 0.96,
        "risk": 0.04,
        "cost": 0.01,
        "latency": 0.1,
        "asset": 0.82,
        "description": "Vector Store v3 — fast semantic similarity search",
    },
    "image-analysis-api": {
        "domain": "vision",
        "reliability": 0.87,
        "risk": 0.09,
        "cost": 0.07,
        "latency": 1.5,
        "asset": 0.79,
        "description": "Image Analysis API — object detection and captioning",
    },
}


# ---------------------------------------------------------------------------
# MarketplaceListing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketplaceListing:
    """Metadata for a candidate tool discovered in the marketplace.

    Attributes
    ----------
    name:
        Unique tool identifier.
    domain:
        Primary domain this tool serves (e.g. ``"finance"``).
    description:
        Short human-readable summary.
    estimated_reliability:
        Provider-stated or community-estimated reliability [0, 1].
    estimated_risk:
        Provider-stated or community-estimated risk [0, 1].
    estimated_cost:
        Estimated cost per call (normalised [0, 1]).
    estimated_latency:
        Estimated latency per call (normalised [0, 1]).
    estimated_asset:
        Estimated value-add (asset score) [0, 1].
    source:
        Where this listing came from (e.g. ``"community_catalog"``,
        ``"huggingface"``, ``"internal"``).
    """

    name: str
    domain: str
    description: str
    estimated_reliability: float
    estimated_risk: float
    estimated_cost: float
    estimated_latency: float
    estimated_asset: float
    source: str = "community_catalog"

    def to_tool_profile(self, *, probationary: bool = True) -> ToolProfile:
        """Convert this listing to a conservative ``ToolProfile``.

        When ``probationary=True`` (default), reliability is discounted by
        25 % to reflect that the tool hasn't been validated yet.

        Parameters
        ----------
        probationary:
            Apply a conservative discount to reliability.

        Returns
        -------
        ToolProfile
        """
        reliability = (
            clamp01(self.estimated_reliability * 0.75)
            if probationary
            else clamp01(self.estimated_reliability)
        )
        return ToolProfile(
            name=self.name,
            cost=clamp01(self.estimated_cost),
            latency=clamp01(self.estimated_latency),
            reliability=reliability,
            risk=clamp01(self.estimated_risk),
            asset=clamp01(self.estimated_asset),
            domain=self.domain,
        )


# ---------------------------------------------------------------------------
# RecruitmentResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecruitmentResult:
    """Outcome of a single recruitment run.

    Attributes
    ----------
    triggered:
        Whether a recruitment search was triggered (``False`` if registry
        was healthy or complexity was below threshold).
    reason:
        Human-readable reason for the result.
    candidates_found:
        Number of marketplace listings found for the task domain.
    selected_listing:
        The listing chosen for on-boarding, or ``None`` if none was suitable.
    scout_pass_rate:
        Fraction of scout tasks that succeeded (0.0 if no scouting occurred).
    registered:
        ``True`` if the new tool was successfully registered.
    registered_tool_name:
        Name of the registered tool, or ``None``.
    """

    triggered: bool
    reason: str
    candidates_found: int
    selected_listing: MarketplaceListing | None
    scout_pass_rate: float
    registered: bool
    registered_tool_name: str | None


# ---------------------------------------------------------------------------
# SovereignRecruiter
# ---------------------------------------------------------------------------


@dataclass
class SovereignRecruiter:
    """Self-expanding capability layer that hires new tools on demand.

    The recruiter monitors the ``ConnectorRegistry``.  When a task exceeds
    the complexity threshold *and* every tool in the registry serving the
    task domain has reliability below the reliability threshold, the recruiter:

    1. Searches the marketplace catalog for candidates in the task domain.
    2. Selects the highest-utility candidate.
    3. Runs ``n_scout_tasks`` probe calls to baseline the candidate.
    4. Registers the candidate as a ``ToolConnector`` tagged ``"probationary"``.

    Parameters
    ----------
    registry:
        The ``ConnectorRegistry`` to monitor and expand.
    complexity_threshold:
        Minimum task complexity to trigger recruitment.  Default: ``0.8``.
    reliability_threshold:
        Maximum acceptable reliability for existing tools — if all tools
        are below this, recruitment is triggered.  Default: ``0.6``.
    n_scout_tasks:
        Number of probe calls to run on the new tool before registration.
        Default: ``3``.
    min_scout_pass_rate:
        Minimum fraction of scout calls that must succeed for registration.
        Default: ``0.5``.
    marketplace:
        Custom tool catalog to search.  Defaults to the built-in catalog.
    probe_fn:
        Callable used for scout tasks.  Defaults to a no-op that always
        returns success.  Replace with a real HTTP probe in production.
    seed:
        Random seed for deterministic scout simulation.

    Example
    -------
    ::

        recruiter = SovereignRecruiter(registry=registry)
        task = BrainTask(prompt="Prove Fermat's Last Theorem", domain="math",
                         complexity=0.95, stakes=0.8)
        result = recruiter.recruit_if_needed(task)
        if result.registered:
            print(f"Hired {result.registered_tool_name!r}")
    """

    registry: ConnectorRegistry
    complexity_threshold: float = 0.8
    reliability_threshold: float = 0.6
    n_scout_tasks: int = 3
    min_scout_pass_rate: float = 0.5
    marketplace: dict[str, dict[str, object]] | None = None
    probe_fn: Callable[[str, str], bool] | None = None
    seed: int = 42

    _recruitment_log: list[RecruitmentResult] = field(
        default_factory=list, init=False, repr=False
    )
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.marketplace is None:
            self.marketplace = _DEFAULT_MARKETPLACE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recruit_if_needed(self, task: BrainTask) -> RecruitmentResult:
        """Check the registry and recruit a new tool if required.

        Parameters
        ----------
        task:
            The task being evaluated.  Complexity and domain are used to
            decide whether recruitment is warranted.

        Returns
        -------
        RecruitmentResult
            Full outcome of the recruitment decision.
        """
        # Check complexity gate
        if task.complexity < self.complexity_threshold:
            result = RecruitmentResult(
                triggered=False,
                reason=f"complexity={task.complexity:.3f} < threshold={self.complexity_threshold:.3f}",
                candidates_found=0,
                selected_listing=None,
                scout_pass_rate=0.0,
                registered=False,
                registered_tool_name=None,
            )
            self._recruitment_log.append(result)
            return result

        # Check if registry has a reliable tool for this domain
        if self._registry_healthy(task.domain):
            result = RecruitmentResult(
                triggered=False,
                reason=f"registry has reliable tool(s) for domain={task.domain!r}",
                candidates_found=0,
                selected_listing=None,
                scout_pass_rate=0.0,
                registered=False,
                registered_tool_name=None,
            )
            self._recruitment_log.append(result)
            return result

        # Trigger discovery
        result = self._discover_and_onboard(task)
        self._recruitment_log.append(result)
        return result

    def recruitment_log(self) -> list[RecruitmentResult]:
        """Return a copy of all recruitment decisions."""
        return list(self._recruitment_log)

    def hired_count(self) -> int:
        """Return the number of tools successfully registered."""
        return sum(1 for r in self._recruitment_log if r.registered)

    def search_marketplace(self, domain: str) -> list[MarketplaceListing]:
        """Search the marketplace catalog for candidates in *domain*.

        Parameters
        ----------
        domain:
            Task domain to search for (e.g. ``"finance"``).  Also returns
            tools tagged ``"general"`` or ``"language"`` as fallbacks.

        Returns
        -------
        list[MarketplaceListing]
            Listings sorted by estimated utility descending.
        """
        catalog = self.marketplace or _DEFAULT_MARKETPLACE
        listings: list[MarketplaceListing] = []
        for name, meta in catalog.items():
            tool_domain = str(meta.get("domain", "general"))
            if tool_domain not in (domain, "general", "language"):
                continue
            # Skip already-registered tools
            if self.registry.get(name) is not None:
                continue
            listing = MarketplaceListing(
                name=name,
                domain=tool_domain,
                description=str(meta.get("description", "")),
                estimated_reliability=float(meta.get("reliability", 0.8)),  # type: ignore[arg-type]
                estimated_risk=float(meta.get("risk", 0.1)),  # type: ignore[arg-type]
                estimated_cost=float(meta.get("cost", 0.05)),  # type: ignore[arg-type]
                estimated_latency=float(meta.get("latency", 0.5)),  # type: ignore[arg-type]
                estimated_asset=float(meta.get("asset", 0.75)),  # type: ignore[arg-type]
                source="community_catalog",
            )
            listings.append(listing)

        # Sort by utility: asset * reliability - cost - latency - risk
        listings.sort(
            key=lambda l: (
                l.estimated_asset * l.estimated_reliability
                - l.estimated_cost
                - l.estimated_latency
                - l.estimated_risk
            ),
            reverse=True,
        )
        return listings

    def run_scout_tasks(self, listing: MarketplaceListing) -> float:
        """Run probe calls against a candidate listing to establish a baseline.

        Parameters
        ----------
        listing:
            The marketplace listing to probe.

        Returns
        -------
        float
            Fraction of scout tasks that succeeded [0, 1].
        """
        if self.n_scout_tasks <= 0:
            return 1.0

        successes = 0
        for _ in range(self.n_scout_tasks):
            if self.probe_fn is not None:
                ok = self.probe_fn(listing.name, listing.domain)
            else:
                # Simulate: use listing reliability as bernoulli probability
                ok = self._rng.random() < listing.estimated_reliability
            if ok:
                successes += 1
        return successes / self.n_scout_tasks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _registry_healthy(self, domain: str) -> bool:
        """Return True if the registry has at least one reliable tool for *domain*."""
        for name in self.registry.names():
            connector = self.registry.get(name)
            if connector is None:
                continue
            profile = connector.refreshed_profile()
            # Match domain or general-purpose tools
            if profile.domain in (domain, "general"):
                if profile.reliability >= self.reliability_threshold:
                    return True
        return False

    def _discover_and_onboard(self, task: BrainTask) -> RecruitmentResult:
        """Internal: search, scout, and register the best candidate."""
        listings = self.search_marketplace(task.domain)
        if not listings:
            return RecruitmentResult(
                triggered=True,
                reason=f"No marketplace candidates found for domain={task.domain!r}",
                candidates_found=0,
                selected_listing=None,
                scout_pass_rate=0.0,
                registered=False,
                registered_tool_name=None,
            )

        # Try candidates in utility order
        for candidate in listings:
            pass_rate = self.run_scout_tasks(candidate)
            if pass_rate < self.min_scout_pass_rate:
                continue  # This candidate failed scouting; try next

            # Build a probationary profile and register
            profile = candidate.to_tool_profile(probationary=True)

            def _make_stub(cname: str) -> Callable[..., dict[str, object]]:  # noqa: ANN001
                def stub(*_args: object, **_kwargs: object) -> dict[str, object]:
                    return {"tool": cname, "status": "probationary_stub"}
                return stub

            connector = ToolConnector(
                name=candidate.name,
                fn=_make_stub(candidate.name),
                profile=profile,
            )
            self.registry.register(connector)

            return RecruitmentResult(
                triggered=True,
                reason=(
                    f"Registry had no reliable tool for domain={task.domain!r}; "
                    f"recruited {candidate.name!r} (scout_pass_rate={pass_rate:.2f})"
                ),
                candidates_found=len(listings),
                selected_listing=candidate,
                scout_pass_rate=pass_rate,
                registered=True,
                registered_tool_name=candidate.name,
            )

        # All candidates failed scouting
        return RecruitmentResult(
            triggered=True,
            reason=(
                f"All {len(listings)} candidate(s) failed scout threshold "
                f"(min_scout_pass_rate={self.min_scout_pass_rate:.2f})"
            ),
            candidates_found=len(listings),
            selected_listing=None,
            scout_pass_rate=0.0,
            registered=False,
            registered_tool_name=None,
        )

    def summary(self) -> dict[str, object]:
        """Return a summary of all recruitment activity.

        Returns
        -------
        dict
            Keys: ``total_checks``, ``triggers``, ``registered``,
            ``not_triggered``, ``hire_rate``.
        """
        log = self._recruitment_log
        triggers = sum(1 for r in log if r.triggered)
        registered = self.hired_count()
        total = len(log)
        return {
            "total_checks": total,
            "triggers": triggers,
            "registered": registered,
            "not_triggered": total - triggers,
            "hire_rate": registered / triggers if triggers > 0 else 0.0,
        }
