"""Phase 41: Genesis Node Bootstrapping — Cold-Start Trust Distribution.

When the very first node in a Swarm boots there are no peers in the
``AgentEconomyLedger`` and no prior trust state.  ``GenesisMint`` solves
this *cold-start* problem by distributing an initial allocation of Trust
Tokens to a set of bootstrap peers according to a *spatial decay* formula:

.. math::

    T_i = \\frac{e^{-\\gamma d_i}}{\\sum_{j=1}^{N} e^{-\\gamma d_j}}

where :math:`d_i` is the network latency/distance to peer *i* and
:math:`\\gamma` is the decay constant.  Closer peers receive a larger
initial allocation.

Key classes
-----------
``GenesisConfig``
    Deterministic configuration for the genesis (root) node.
``GenesisAllocation``
    Immutable record of a single peer's initial token grant.
``GenesisMint``
    Computes and records the genesis token distribution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# GenesisConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenesisConfig:
    """Deterministic configuration for the genesis (root) node.

    Parameters
    ----------
    total_tokens:
        Total trust tokens to distribute across all bootstrap peers.
        Default: ``1000.0``.
    gamma:
        Decay constant γ for the spatial-decay formula.  Higher values
        concentrate the distribution near close peers.  Default: ``1.0``.
    genesis_node_id:
        Identifier for the genesis node itself.  Default: ``"genesis-0"``.
    initial_threshold:
        Starting interceptor risk-veto threshold for the genesis node.
        Default: ``0.45``.
    """

    total_tokens: float = 1000.0
    gamma: float = 1.0
    genesis_node_id: str = "genesis-0"
    initial_threshold: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "total_tokens": self.total_tokens,
            "gamma": self.gamma,
            "genesis_node_id": self.genesis_node_id,
            "initial_threshold": self.initial_threshold,
        }


# ---------------------------------------------------------------------------
# GenesisAllocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenesisAllocation:
    """Immutable record of a single peer's genesis token allocation.

    Attributes
    ----------
    peer_id:
        Peer identifier.
    distance:
        Network latency/distance used in the decay formula.
    weight:
        Un-normalised weight :math:`e^{-\\gamma d}`.
    tokens:
        Allocated tokens (``total_tokens * normalised_weight``).
    """

    peer_id: str
    distance: float
    weight: float
    tokens: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "peer_id": self.peer_id,
            "distance": self.distance,
            "weight": self.weight,
            "tokens": self.tokens,
        }


# ---------------------------------------------------------------------------
# GenesisMint
# ---------------------------------------------------------------------------


@dataclass
class GenesisMint:
    """Computes the genesis token distribution using spatial decay.

    Call :meth:`mint` with a mapping of peer IDs to distances to obtain
    a list of :class:`GenesisAllocation` objects.

    Parameters
    ----------
    config:
        Genesis configuration.

    Example
    -------
    ::

        mint = GenesisMint(GenesisConfig(total_tokens=500.0, gamma=0.5))
        allocs = mint.mint({"peer-a": 1.0, "peer-b": 3.0})
        for a in allocs:
            print(a.peer_id, a.tokens)
    """

    config: GenesisConfig = field(default_factory=GenesisConfig)

    _history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def mint(self, peer_distances: dict[str, float]) -> list[GenesisAllocation]:
        """Distribute genesis tokens across *peer_distances*.

        If *peer_distances* is empty, an empty list is returned and no
        tokens are allocated.

        Parameters
        ----------
        peer_distances:
            Mapping of ``{peer_id: distance}`` where distance is a
            non-negative float representing network latency.

        Returns
        -------
        list[GenesisAllocation]
            One allocation per peer, sorted by descending token grant.

        Raises
        ------
        ValueError
            If any distance value is negative.
        """
        if not peer_distances:
            return []

        for peer_id, dist in peer_distances.items():
            if dist < 0:
                raise ValueError(
                    f"Distance for peer {peer_id!r} must be non-negative, got {dist}"
                )

        gamma = self.config.gamma
        weights: dict[str, float] = {
            peer_id: math.exp(-gamma * dist)
            for peer_id, dist in peer_distances.items()
        }
        total_weight = sum(weights.values())
        if total_weight == 0.0:
            total_weight = 1.0  # safety guard

        allocations: list[GenesisAllocation] = [
            GenesisAllocation(
                peer_id=pid,
                distance=peer_distances[pid],
                weight=w,
                tokens=self.config.total_tokens * (w / total_weight),
            )
            for pid, w in weights.items()
        ]
        allocations.sort(key=lambda a: -a.tokens)

        record: dict[str, Any] = {
            "timestamp": time.time(),
            "total_tokens": self.config.total_tokens,
            "gamma": gamma,
            "peer_count": len(allocations),
            "allocations": [a.to_dict() for a in allocations],
        }
        self._history.append(record)
        return allocations

    def history(self) -> list[dict[str, Any]]:
        """Return a copy of all previous mint events."""
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of genesis activity."""
        return {
            "genesis_node_id": self.config.genesis_node_id,
            "total_tokens": self.config.total_tokens,
            "gamma": self.config.gamma,
            "mint_events": len(self._history),
        }
