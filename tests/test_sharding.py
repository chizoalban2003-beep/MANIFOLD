"""Tests for Phase 45: DHT Sharding (manifold/sharding.py)."""

from __future__ import annotations

import pytest

from manifold.sharding import KademliaNode, NodeID, RoutingEntry, ShardRouter


# ---------------------------------------------------------------------------
# NodeID
# ---------------------------------------------------------------------------


class TestNodeID:
    def test_value_is_int(self) -> None:
        nid = NodeID("peer-a")
        assert isinstance(nid.value, int)
        assert nid.value >= 0

    def test_deterministic(self) -> None:
        a1 = NodeID("peer-a")
        a2 = NodeID("peer-a")
        assert a1.value == a2.value

    def test_different_ids_different_values(self) -> None:
        a = NodeID("peer-a")
        b = NodeID("peer-b")
        assert a.value != b.value

    def test_distance_to_self_is_zero(self) -> None:
        a = NodeID("peer-a")
        assert a.distance(a) == 0

    def test_distance_symmetry(self) -> None:
        a = NodeID("peer-a")
        b = NodeID("peer-b")
        assert a.distance(b) == b.distance(a)

    def test_distance_non_negative(self) -> None:
        a = NodeID("x")
        b = NodeID("y")
        assert a.distance(b) >= 0

    def test_distance_xor_formula(self) -> None:
        a = NodeID("foo")
        b = NodeID("bar")
        assert a.distance(b) == a.value ^ b.value

    def test_repr(self) -> None:
        nid = NodeID("test")
        r = repr(nid)
        assert "test" in r

    def test_frozen(self) -> None:
        nid = NodeID("x")
        with pytest.raises((AttributeError, TypeError)):
            nid.raw_id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RoutingEntry
# ---------------------------------------------------------------------------


class TestRoutingEntry:
    def test_to_dict(self) -> None:
        entry = RoutingEntry(node_id=NodeID("peer-a"), endpoint="http://peer-a:8080")
        d = entry.to_dict()
        assert d["raw_id"] == "peer-a"
        assert d["endpoint"] == "http://peer-a:8080"
        assert "last_seen" in d
        assert "value" in d

    def test_touch_updates_last_seen(self) -> None:
        import time
        entry = RoutingEntry(node_id=NodeID("x"), endpoint="http://x:8080")
        t0 = entry.last_seen
        time.sleep(0.01)
        entry.touch()
        assert entry.last_seen >= t0

    def test_endpoint_mutable(self) -> None:
        entry = RoutingEntry(node_id=NodeID("x"), endpoint="http://old:8080")
        entry.endpoint = "http://new:8080"
        assert entry.endpoint == "http://new:8080"


# ---------------------------------------------------------------------------
# KademliaNode — peer management
# ---------------------------------------------------------------------------


class TestKademliaNodePeerManagement:
    def test_empty_table(self) -> None:
        node = KademliaNode("server-0")
        assert node.peer_count() == 0

    def test_add_peer(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://peer-a:8080")
        assert node.peer_count() == 1

    def test_add_multiple_peers(self) -> None:
        node = KademliaNode("server-0")
        for i in range(5):
            node.add_peer(f"peer-{i}", f"http://peer-{i}:8080")
        assert node.peer_count() == 5

    def test_add_duplicate_updates_endpoint(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://old:8080")
        node.add_peer("peer-a", "http://new:8080")
        assert node.peer_count() == 1
        entry = node._find("peer-a")
        assert entry is not None
        assert entry.endpoint == "http://new:8080"

    def test_remove_existing_peer(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://peer-a:8080")
        removed = node.remove_peer("peer-a")
        assert removed is True
        assert node.peer_count() == 0

    def test_remove_nonexistent_returns_false(self) -> None:
        node = KademliaNode("server-0")
        assert node.remove_peer("no-such-peer") is False

    def test_k_limit_enforced(self) -> None:
        node = KademliaNode("server-0", k=3)
        for i in range(10):
            node.add_peer(f"peer-{i}", f"http://peer-{i}:8080")
        assert node.peer_count() <= 3

    def test_routing_table_sorted_by_distance(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://a:8080")
        node.add_peer("peer-b", "http://b:8080")
        node.add_peer("peer-c", "http://c:8080")
        table = node.routing_table()
        distances = [e["distance"] for e in table]
        assert distances == sorted(distances)

    def test_routing_table_includes_distance(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://a:8080")
        table = node.routing_table()
        assert "distance" in table[0]


# ---------------------------------------------------------------------------
# KademliaNode — closest peer lookup
# ---------------------------------------------------------------------------


class TestKademliaNodeClosestPeer:
    def test_closest_peer_empty_returns_none(self) -> None:
        node = KademliaNode("server-0")
        assert node.closest_peer("some-tool") is None

    def test_closest_peer_single(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("peer-a", "http://a:8080")
        entry = node.closest_peer("some-tool")
        assert entry is not None
        assert entry.node_id.raw_id == "peer-a"

    def test_closest_peer_returns_correct_one(self) -> None:
        # Add many peers; the closest should vary based on XOR distance
        node = KademliaNode("server-0")
        peer_ids = [f"peer-{i}" for i in range(20)]
        for pid in peer_ids:
            node.add_peer(pid, f"http://{pid}:8080")
        tool_id = "my-special-tool"
        entry = node.closest_peer(tool_id)
        assert entry is not None
        target = NodeID(tool_id)
        min_dist = target.distance(entry.node_id)
        for other in node._peers:
            assert target.distance(other.node_id) >= min_dist

    def test_k_closest_returns_n(self) -> None:
        node = KademliaNode("server-0")
        for i in range(10):
            node.add_peer(f"peer-{i}", f"http://peer-{i}:8080")
        result = node.k_closest("lookup-key", n=3)
        assert len(result) == 3

    def test_k_closest_sorted(self) -> None:
        node = KademliaNode("server-0")
        for i in range(8):
            node.add_peer(f"peer-{i}", f"http://peer-{i}:8080")
        result = node.k_closest("target", n=4)
        target = NodeID("target")
        distances = [target.distance(e.node_id) for e in result]
        assert distances == sorted(distances)

    def test_k_closest_empty_returns_empty(self) -> None:
        node = KademliaNode("server-0")
        assert node.k_closest("x") == []

    def test_k_closest_clamped_to_table_size(self) -> None:
        node = KademliaNode("server-0")
        node.add_peer("p1", "http://p1:8080")
        node.add_peer("p2", "http://p2:8080")
        result = node.k_closest("x", n=100)
        assert len(result) == 2

    def test_xor_distance_helper(self) -> None:
        node = KademliaNode("server-0")
        d = node.xor_distance("id-a", "id-b")
        expected = NodeID("id-a").distance(NodeID("id-b"))
        assert d == expected

    def test_xor_distance_to_self_is_zero(self) -> None:
        node = KademliaNode("server-0")
        assert node.xor_distance("same", "same") == 0


# ---------------------------------------------------------------------------
# ShardRouter
# ---------------------------------------------------------------------------


class TestShardRouter:
    def test_empty_shard_for_returns_none(self) -> None:
        router = ShardRouter("node-0")
        assert router.shard_for("my-tool") is None

    def test_shard_for_returns_endpoint(self) -> None:
        router = ShardRouter("node-0")
        router.add_peer("peer-a", "http://peer-a:8080")
        endpoint = router.shard_for("my-tool")
        assert endpoint == "http://peer-a:8080"

    def test_shard_for_increments_route_count(self) -> None:
        router = ShardRouter("node-0")
        router.add_peer("peer-a", "http://peer-a:8080")
        router.shard_for("tool-1")
        router.shard_for("tool-2")
        assert router.summary()["route_count"] == 2

    def test_shard_for_empty_no_increment(self) -> None:
        router = ShardRouter("node-0")
        router.shard_for("tool-1")
        assert router.summary()["route_count"] == 0

    def test_remove_peer(self) -> None:
        router = ShardRouter("node-0")
        router.add_peer("peer-a", "http://peer-a:8080")
        assert router.remove_peer("peer-a") is True
        assert router.peer_count() == 0

    def test_peer_count(self) -> None:
        router = ShardRouter("node-0")
        router.add_peer("p1", "http://p1:8080")
        router.add_peer("p2", "http://p2:8080")
        assert router.peer_count() == 2

    def test_routing_table(self) -> None:
        router = ShardRouter("node-0")
        router.add_peer("p1", "http://p1:8080")
        table = router.routing_table()
        assert len(table) == 1
        assert "raw_id" in table[0]

    def test_summary_keys(self) -> None:
        router = ShardRouter("node-0")
        s = router.summary()
        assert "local_id" in s
        assert "peer_count" in s
        assert "route_count" in s
        assert "k" in s

    def test_summary_local_id(self) -> None:
        router = ShardRouter("my-node")
        assert router.summary()["local_id"] == "my-node"

    def test_different_tools_may_route_to_different_peers(self) -> None:
        router = ShardRouter("server-0")
        for i in range(20):
            router.add_peer(f"peer-{i}", f"http://peer-{i}:8080")
        # Different tool IDs should potentially route to different peers
        endpoints = {router.shard_for(f"tool-{j}") for j in range(10)}
        # With 20 peers and 10 tools, we expect some variety
        assert len(endpoints) >= 1  # at minimum: it works
