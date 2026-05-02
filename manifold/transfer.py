"""Phase 6: Cross-Domain Reputation Transfer — the Warm Start Problem.

When a ``ManifoldBrain`` is deployed in a new domain, it currently starts as
a *tabula rasa*: all tools are assumed reliable until proved otherwise.  This
means the first 10-30 interactions are wasted "rediscovering" known-bad tools.

``ReputationRegistry`` solves this by maintaining a **global aggregate** of
tool success rates observed across *all* domains.  When a new brain is spun
up for an unfamiliar domain, its ``BrainMemory`` can be warm-started using:

    Rep_0 = α × Rep_Global + (1-α) × Rep_Default

where:
- ``Rep_Global`` is the aggregate success rate from the registry.
- ``Rep_Default`` is the tool's stated reliability (from ``ToolProfile``).
- ``α`` is the **transferability coefficient** — high (0.8) for related domains,
  low (0.3) for disparate domains.

This prevents a "Social Reset" every time a new task appears, while still
allowing for domain-specific redemption: a tool that is a liar in "Customer
Support" starts with a reputation handicap in "Legal Research" (α=0.3), but
can fully recover if it performs well there.

Key classes
-----------
``ReputationRegistry``
    Aggregates tool success rates from ``BrainMemory`` or ``GossipBus``
    observations across multiple agents/domains.
``WarmStartConfig``
    Domain pair transferability coefficients.
``warm_start_memory``
    Factory function: create a ``BrainMemory`` pre-populated with
    warm-started tool stats from a ``ReputationRegistry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .brain import BrainMemory, ToolProfile
from .trustrouter import clamp01


@dataclass
class ReputationRegistry:
    """Global tool reputation store aggregated across agents and domains.

    Tool success rates observed in any domain contribute to the registry.
    The registry acts as the "institutional memory" of the MANIFOLD ecosystem:
    it answers "how has tool X performed across all domains, historically?"

    Parameters
    ----------
    decay_rate:
        Per-update decay applied to existing aggregate stats when new
        observations arrive.  Lower = longer memory.  Defaults to 0.05.

    Example
    -------
    ::

        registry = ReputationRegistry()
        # Observe outcomes from a deployed brain:
        registry.observe("web_search", success_rate=0.72, n_observations=15)
        registry.observe("legal_search", success_rate=0.45, n_observations=8)

        # Warm-start a new brain for a related domain:
        memory = warm_start_memory(registry, alpha=0.7)
    """

    decay_rate: float = 0.05

    _stats: dict[str, dict[str, float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def observe(
        self,
        tool_name: str,
        success_rate: float,
        n_observations: int = 1,
    ) -> None:
        """Record a batch of tool observations into the global aggregate.

        Parameters
        ----------
        tool_name:
            The tool identifier (matches ``ToolProfile.name``).
        success_rate:
            Observed success rate for this batch, in [0, 1].
        n_observations:
            Number of observations in this batch.  More observations = higher
            weight in the running aggregate.
        """
        current = self._stats.setdefault(
            tool_name,
            {"success_rate": 1.0, "count": 0.0},
        )
        old_count = current["count"]
        new_count = old_count + n_observations
        # Running weighted average (Welford-style update)
        current["success_rate"] = (
            current["success_rate"] * old_count + success_rate * n_observations
        ) / max(1.0, new_count)
        current["count"] = new_count

    def observe_from_memory(self, memory: BrainMemory) -> None:
        """Ingest all tool stats from a ``BrainMemory`` instance into the registry.

        This is the primary way to keep the registry updated: after each
        ``LiveBrain.learn()`` cycle, call this method so the global registry
        reflects the latest local knowledge.

        Parameters
        ----------
        memory:
            A ``BrainMemory`` whose ``tool_stats`` are absorbed.
        """
        for tool_name, stats in memory.tool_stats.items():
            sr = stats.get("success_rate", 1.0)
            n = max(1, int(stats.get("count", 1.0)))
            self.observe(tool_name, success_rate=sr, n_observations=n)

    def global_success_rate(self, tool_name: str) -> float | None:
        """Return the aggregate success rate for *tool_name*, or ``None`` if unknown.

        Parameters
        ----------
        tool_name:
            The tool identifier.
        """
        entry = self._stats.get(tool_name)
        if entry is None:
            return None
        return entry["success_rate"]

    def all_rates(self) -> dict[str, float]:
        """Return a copy of all (tool → success_rate) pairs in the registry."""
        return {name: s["success_rate"] for name, s in self._stats.items()}

    def observation_count(self, tool_name: str) -> int:
        """Return the total observation count for *tool_name* (0 if unknown)."""
        return int(self._stats.get(tool_name, {}).get("count", 0))


@dataclass(frozen=True)
class WarmStartConfig:
    """Transferability coefficients between domain pairs.

    ``alpha(source_domain, target_domain)`` returns the fraction of global
    reputation that flows into the warm start.  High α (0.8) for related
    domains; low α (0.3) for disparate ones.

    Parameters
    ----------
    related_domains:
        Set of (source, target) domain pairs considered related.
        Symmetric by default.
    related_alpha:
        Transferability coefficient for related domain pairs.
    unrelated_alpha:
        Transferability coefficient for unrelated domain pairs.

    Example
    -------
    ::

        cfg = WarmStartConfig(
            related_domains={("support", "billing"), ("legal", "compliance")},
            related_alpha=0.8,
            unrelated_alpha=0.3,
        )
        alpha = cfg.alpha("support", "billing")  # → 0.8
        alpha = cfg.alpha("support", "coding")   # → 0.3
    """

    related_domains: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    related_alpha: float = 0.8
    unrelated_alpha: float = 0.3

    def alpha(self, source_domain: str, target_domain: str) -> float:
        """Return the transferability coefficient for (source → target)."""
        pair = (source_domain, target_domain)
        pair_rev = (target_domain, source_domain)
        if pair in self.related_domains or pair_rev in self.related_domains:
            return self.related_alpha
        return self.unrelated_alpha


def warm_start_memory(
    registry: ReputationRegistry,
    tools: list[ToolProfile],
    alpha: float = 0.5,
    base_learning_rate: float = 0.15,
) -> BrainMemory:
    """Create a ``BrainMemory`` pre-populated with warm-started tool stats.

    Applies the reputation transfer formula for each known tool:

        Rep_0 = α × Rep_Global + (1-α) × Rep_Default

    Tools not in the registry are left at their stated reliability (pure
    prior), so the warm start is always safe to apply.

    Parameters
    ----------
    registry:
        The ``ReputationRegistry`` holding global aggregate stats.
    tools:
        List of ``ToolProfile`` instances whose reliability priors are used
        as ``Rep_Default``.
    alpha:
        Transferability coefficient in [0, 1].
        - 0.0 = ignore global reputation entirely (cold start)
        - 1.0 = fully trust global reputation (aggressive warm start)
        - 0.5 = balanced blend (default)
    base_learning_rate:
        Passed through to the new ``BrainMemory``.

    Returns
    -------
    BrainMemory
        A new memory instance with tool_stats pre-seeded.  The ``count``
        for each warm-started tool is set to the registry observation count,
        so the brain knows how confident the prior is.

    Example
    -------
    ::

        registry = ReputationRegistry()
        registry.observe("web_search", success_rate=0.65, n_observations=20)
        tools = default_tools()

        # New brain for "legal" domain, related to "support" (alpha=0.7)
        memory = warm_start_memory(registry, tools, alpha=0.7)
        brain = ManifoldBrain(config, tools=tools, memory=memory)
    """
    alpha = clamp01(alpha)
    memory = BrainMemory(base_learning_rate=base_learning_rate)
    for tool in tools:
        global_rate = registry.global_success_rate(tool.name)
        if global_rate is None:
            continue  # tool not observed globally yet — cold start for this tool
        rep_0 = clamp01(alpha * global_rate + (1.0 - alpha) * tool.reliability)
        n_obs = registry.observation_count(tool.name)
        memory.tool_stats[tool.name] = {
            "success_rate": rep_0,
            "count": float(n_obs),
            "utility": tool.utility,
            "consecutive_failures": 0.0,
        }
    return memory
