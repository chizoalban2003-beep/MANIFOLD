"""Phase 45: Distributed Hash Table (DHT) Sharding — XOR-Metric Routing.

As the Swarm scales a single node cannot hold the entire
``GlobalReputationLedger``.  ``sharding.py`` implements a lightweight,
pure-Python Kademlia-inspired routing table.

Distance metric
---------------
.. math::

    Distance(A, B) = Hash(A) \\oplus Hash(B)

where the "hash" is the integer representation of the first 8 bytes of
:func:`hashlib.sha256` applied to the node / key identifier.

Reputation Sharding
-------------------
When a node receives a :class:`~manifold.federation.FederatedGossipPacket`
for a specific Tool ID it calls :meth:`KademliaNode.closest_peer` to find
the peer closest (smallest XOR distance) to that Tool ID, instead of
broadcasting the packet to every known peer.

Key classes
-----------
``NodeID``
    Thin wrapper for a 64-bit XOR-comparable node identity.
``RoutingEntry``
    A single entry in the k-bucket routing table.
``KademliaNode``
    XOR-metric routing table + closest-peer lookup.
``ShardRouter``
    Higher-level helper that wraps a :class:`~manifold.swarm.SwarmRouter`
    and routes :class:`~manifold.federation.FederatedGossipPacket` objects
    to the topologically nearest peer.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# NodeID
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeID:
    """A 64-bit XOR-comparable node identity derived from a string identifier.

    Parameters
    ----------
    raw_id:
        Human-readable identifier (e.g. ``"peer-a"`` or a tool name).

    Attributes
    ----------
    value:
        Unsigned 64-bit integer computed as ``sha256(raw_id)[:8]``.
    """

    raw_id: str

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.raw_id.encode("utf-8")).digest()
        # store as attribute bypassing frozen restriction via object.__setattr__
        object.__setattr__(self, "value", struct.unpack(">Q", digest[:8])[0])

    # mypy hint
    value: int = field(default=0, init=False)

    def distance(self, other: "NodeID") -> int:
        """XOR distance between this node and *other*.

        Parameters
        ----------
        other:
            The other node.

        Returns
        -------
        int
            Non-negative XOR distance.
        """
        return self.value ^ other.value

    def __repr__(self) -> str:
        return f"NodeID({self.raw_id!r}, value={self.value:016x})"


# ---------------------------------------------------------------------------
# RoutingEntry
# ---------------------------------------------------------------------------


@dataclass
class RoutingEntry:
    """A single entry in the k-bucket routing table.

    Attributes
    ----------
    node_id:
        The :class:`NodeID` of the peer.
    endpoint:
        Base URL of the peer's MANIFOLD server.
    last_seen:
        POSIX timestamp of the last successful contact.
    """

    node_id: NodeID
    endpoint: str
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "raw_id": self.node_id.raw_id,
            "value": self.node_id.value,
            "endpoint": self.endpoint,
            "last_seen": self.last_seen,
        }

    def touch(self) -> None:
        """Update ``last_seen`` to the current time."""
        self.last_seen = time.time()


# ---------------------------------------------------------------------------
# KademliaNode
# ---------------------------------------------------------------------------


@dataclass
class KademliaNode:
    """A lightweight XOR-metric Kademlia-inspired routing table.

    The node maintains a flat list of known peers (no full k-bucket tree;
    sufficient for MANIFOLD's bounded Swarm size).

    Parameters
    ----------
    local_id:
        Identifier for *this* node.  All distances are measured relative
        to this ID.
    k:
        Maximum number of entries to keep per routing table refresh.
        Default: ``20``.

    Example
    -------
    ::

        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://peer-a:8080")
        entry = node.closest_peer("my-tool")
        print(entry.endpoint)
    """

    local_id: str
    k: int = 20

    _local_node_id: NodeID = field(init=False, repr=False)
    _peers: list[RoutingEntry] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._local_node_id = NodeID(self.local_id)

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, peer_id: str, endpoint: str) -> None:
        """Add or refresh a peer in the routing table.

        If the peer already exists its ``last_seen`` timestamp is updated.
        If the table would exceed *k* entries, the oldest entry (by
        ``last_seen``) is evicted.

        Parameters
        ----------
        peer_id:
            Unique identifier for the peer.
        endpoint:
            Base URL of the peer's server.
        """
        existing = self._find(peer_id)
        if existing is not None:
            existing.endpoint = endpoint
            existing.touch()
            return

        entry = RoutingEntry(node_id=NodeID(peer_id), endpoint=endpoint)
        self._peers.append(entry)
        if len(self._peers) > self.k:
            # Evict the oldest peer
            self._peers.sort(key=lambda e: e.last_seen, reverse=True)
            self._peers = self._peers[: self.k]

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from the routing table.

        Parameters
        ----------
        peer_id:
            Identifier of the peer to remove.

        Returns
        -------
        bool
            ``True`` if the peer was present and removed.
        """
        before = len(self._peers)
        self._peers = [e for e in self._peers if e.node_id.raw_id != peer_id]
        return len(self._peers) < before

    def peer_count(self) -> int:
        """Return the number of entries in the routing table."""
        return len(self._peers)

    def routing_table(self) -> list[dict[str, Any]]:
        """Return all routing entries sorted by ascending XOR distance.

        Returns
        -------
        list[dict[str, Any]]
            Each entry includes ``raw_id``, ``value``, ``endpoint``,
            ``last_seen``, and ``distance``.
        """
        local = self._local_node_id
        result = []
        for entry in self._peers:
            d = entry.to_dict()
            d["distance"] = local.distance(entry.node_id)
            result.append(d)
        result.sort(key=lambda d: d["distance"])
        return result

    # ------------------------------------------------------------------
    # Closest-peer lookup
    # ------------------------------------------------------------------

    def closest_peer(self, key: str) -> RoutingEntry | None:
        """Return the peer whose :class:`NodeID` is XOR-closest to *key*.

        Parameters
        ----------
        key:
            The lookup key (e.g. a Tool ID from a
            :class:`~manifold.federation.FederatedGossipPacket`).

        Returns
        -------
        RoutingEntry | None
            The closest peer, or ``None`` if the routing table is empty.
        """
        if not self._peers:
            return None
        target = NodeID(key)
        return min(self._peers, key=lambda e: target.distance(e.node_id))

    def k_closest(self, key: str, n: int = 3) -> list[RoutingEntry]:
        """Return the *n* peers XOR-closest to *key*, sorted ascending.

        Parameters
        ----------
        key:
            The lookup key.
        n:
            How many peers to return.  Clamped to the table size.

        Returns
        -------
        list[RoutingEntry]
        """
        if not self._peers:
            return []
        target = NodeID(key)
        sorted_peers = sorted(self._peers, key=lambda e: target.distance(e.node_id))
        return sorted_peers[: min(n, len(sorted_peers))]

    def xor_distance(self, id_a: str, id_b: str) -> int:
        """Compute the XOR distance between two arbitrary IDs.

        Parameters
        ----------
        id_a:
            First identifier.
        id_b:
            Second identifier.

        Returns
        -------
        int
            Non-negative XOR distance.
        """
        return NodeID(id_a).distance(NodeID(id_b))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find(self, peer_id: str) -> RoutingEntry | None:
        for entry in self._peers:
            if entry.node_id.raw_id == peer_id:
                return entry
        return None


# ---------------------------------------------------------------------------
# ShardRouter
# ---------------------------------------------------------------------------


@dataclass
class ShardRouter:
    """Routes :class:`~manifold.federation.FederatedGossipPacket` objects
    to the topologically nearest peer using XOR distance.

    Instead of broadcasting every gossip packet to all known peers, the
    :class:`ShardRouter` uses its :class:`KademliaNode` to select only the
    closest peer for a given Tool ID.

    Parameters
    ----------
    local_id:
        Identifier for the local node (same value passed to
        :class:`KademliaNode`).
    k:
        Routing-table size limit.

    Example
    -------
    ::

        router = ShardRouter("node-0")
        router.add_peer("peer-a", "http://peer-a:8080")
        endpoint = router.shard_for("my-tool")
        print(endpoint)
    """

    local_id: str
    k: int = 20

    _dht: KademliaNode = field(init=False, repr=False)
    _route_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._dht = KademliaNode(local_id=self.local_id, k=self.k)

    def add_peer(self, peer_id: str, endpoint: str) -> None:
        """Register a peer in the underlying DHT."""
        self._dht.add_peer(peer_id, endpoint)

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from the underlying DHT."""
        return self._dht.remove_peer(peer_id)

    def shard_for(self, tool_id: str) -> str | None:
        """Return the endpoint of the peer closest to *tool_id*.

        Parameters
        ----------
        tool_id:
            The Tool ID from a :class:`~manifold.federation.FederatedGossipPacket`.

        Returns
        -------
        str | None
            The endpoint URL of the nearest peer, or ``None`` if the
            routing table is empty.
        """
        entry = self._dht.closest_peer(tool_id)
        if entry is None:
            return None
        self._route_count += 1
        return entry.endpoint

    def peer_count(self) -> int:
        """Return the number of peers in the DHT routing table."""
        return self._dht.peer_count()

    def routing_table(self) -> list[dict[str, Any]]:
        """Return the DHT routing table sorted by XOR distance."""
        return self._dht.routing_table()

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of shard routing activity."""
        return {
            "local_id": self.local_id,
            "peer_count": self._dht.peer_count(),
            "route_count": self._route_count,
            "k": self.k,
        }
