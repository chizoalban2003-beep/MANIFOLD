"""Tests for Phase 33: Mesh Swarm Routing (manifold/swarm.py)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import pytest

from manifold.brain import BrainTask
from manifold.swarm import SwarmPeer, SwarmRouteResult, SwarmRouter


# ---------------------------------------------------------------------------
# SwarmPeer
# ---------------------------------------------------------------------------


class TestSwarmPeer:
    def test_basic_creation(self) -> None:
        peer = SwarmPeer(org_id="peer-a", endpoint="http://peer-a:8080")
        assert peer.org_id == "peer-a"
        assert peer.endpoint == "http://peer-a:8080"
        assert peer.asset == 1.0
        assert peer.risk == 0.1
        assert peer.cost_routing == 0.1

    def test_custom_attributes(self) -> None:
        peer = SwarmPeer(
            org_id="peer-b",
            endpoint="http://peer-b:9090",
            asset=0.8,
            risk=0.3,
            cost_routing=0.2,
            token_availability=0.6,
        )
        assert peer.asset == 0.8
        assert peer.risk == 0.3
        assert peer.token_availability == 0.6

    def test_routing_value_formula(self) -> None:
        peer = SwarmPeer(
            org_id="p",
            endpoint="http://p:80",
            asset=1.0,
            risk=0.1,
            cost_routing=0.1,
            token_availability=1.0,
        )
        # V = (1.0 + 1.0)/2 - 0.1 - 0.1 = 0.8
        assert peer.routing_value == pytest.approx(0.8)

    def test_routing_value_high_risk_negative(self) -> None:
        peer = SwarmPeer(
            org_id="bad",
            endpoint="http://bad:80",
            asset=0.2,
            risk=0.9,
            cost_routing=0.5,
            token_availability=0.2,
        )
        # clamped to [-1, 1]
        assert peer.routing_value <= 0.0

    def test_routing_value_clamped(self) -> None:
        peer = SwarmPeer("p", "http://p:80", asset=100.0, risk=0.0, cost_routing=0.0)
        assert peer.routing_value <= 1.0

    def test_routing_value_clamped_min(self) -> None:
        peer = SwarmPeer("p", "http://p:80", asset=0.0, risk=1.0, cost_routing=1.0)
        assert peer.routing_value >= -1.0

    def test_shield_url(self) -> None:
        peer = SwarmPeer("p", "http://peer:8080/")
        assert peer.shield_url() == "http://peer:8080/shield"

    def test_shield_url_no_trailing_slash(self) -> None:
        peer = SwarmPeer("p", "http://peer:8080")
        assert peer.shield_url() == "http://peer:8080/shield"

    def test_to_dict(self) -> None:
        peer = SwarmPeer("p", "http://p:80")
        d = peer.to_dict()
        assert d["org_id"] == "p"
        assert "routing_value" in d
        assert "endpoint" in d

    def test_frozen(self) -> None:
        peer = SwarmPeer("p", "http://p:80")
        with pytest.raises((AttributeError, TypeError)):
            peer.org_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SwarmRouteResult
# ---------------------------------------------------------------------------


class TestSwarmRouteResult:
    def test_not_delegated(self) -> None:
        result = SwarmRouteResult(
            delegated=False,
            peer=None,
            routing_value=0.0,
            response=None,
            error="No peers",
        )
        assert result.delegated is False
        assert result.peer is None

    def test_delegated_result(self) -> None:
        peer = SwarmPeer("p", "http://p:80")
        result = SwarmRouteResult(
            delegated=True,
            peer=peer,
            routing_value=0.8,
            response={"vetoed": False},
            error="",
        )
        assert result.delegated is True
        assert result.response == {"vetoed": False}

    def test_to_dict_not_delegated(self) -> None:
        result = SwarmRouteResult(False, None, 0.0, None, "err")
        d = result.to_dict()
        assert d["delegated"] is False
        assert d["peer"] is None
        assert d["error"] == "err"

    def test_to_dict_delegated(self) -> None:
        peer = SwarmPeer("p", "http://p:80")
        result = SwarmRouteResult(True, peer, 0.9, {"ok": True}, "")
        d = result.to_dict()
        assert d["peer"]["org_id"] == "p"

    def test_frozen(self) -> None:
        result = SwarmRouteResult(False, None, 0.0, None, "")
        with pytest.raises((AttributeError, TypeError)):
            result.delegated = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SwarmRouter
# ---------------------------------------------------------------------------


class TestSwarmRouterRegistration:
    def test_empty_router(self) -> None:
        router = SwarmRouter()
        assert router.peer_count() == 0

    def test_register_peer(self) -> None:
        router = SwarmRouter()
        peer = SwarmPeer("peer-a", "http://peer-a:80")
        router.register_peer(peer)
        assert router.peer_count() == 1

    def test_register_replaces_same_org_id(self) -> None:
        router = SwarmRouter()
        router.register_peer(SwarmPeer("peer-a", "http://old:80"))
        router.register_peer(SwarmPeer("peer-a", "http://new:80"))
        assert router.peer_count() == 1
        assert router.peers[0].endpoint == "http://new:80"

    def test_register_multiple_peers(self) -> None:
        router = SwarmRouter()
        router.register_peer(SwarmPeer("a", "http://a:80"))
        router.register_peer(SwarmPeer("b", "http://b:80"))
        assert router.peer_count() == 2

    def test_remove_peer_existing(self) -> None:
        router = SwarmRouter()
        router.register_peer(SwarmPeer("a", "http://a:80"))
        removed = router.remove_peer("a")
        assert removed is True
        assert router.peer_count() == 0

    def test_remove_peer_nonexistent(self) -> None:
        router = SwarmRouter()
        removed = router.remove_peer("nonexistent")
        assert removed is False


class TestSwarmRouterBestPeer:
    def test_no_peers(self) -> None:
        router = SwarmRouter()
        assert router.best_peer() is None

    def test_single_peer(self) -> None:
        router = SwarmRouter()
        peer = SwarmPeer("a", "http://a:80", asset=1.0, risk=0.1, cost_routing=0.1)
        router.register_peer(peer)
        assert router.best_peer() == peer

    def test_picks_highest_value(self) -> None:
        router = SwarmRouter()
        router.register_peer(SwarmPeer("low", "http://low:80", asset=0.2, risk=0.5))
        router.register_peer(SwarmPeer("high", "http://high:80", asset=1.0, risk=0.1))
        best = router.best_peer()
        assert best is not None
        assert best.org_id == "high"

    def test_min_routing_value_filter(self) -> None:
        router = SwarmRouter(min_routing_value=0.5)
        router.register_peer(SwarmPeer("poor", "http://poor:80", asset=0.1, risk=0.9))
        assert router.best_peer() is None

    def test_min_routing_value_allows_good_peer(self) -> None:
        router = SwarmRouter(min_routing_value=0.5)
        router.register_peer(SwarmPeer("good", "http://good:80", asset=1.0, risk=0.0, cost_routing=0.0))
        assert router.best_peer() is not None


class TestSwarmRouterRouteNoPeers:
    def test_no_peers_returns_not_delegated(self) -> None:
        router = SwarmRouter()
        task = BrainTask(prompt="test", domain="general")
        result = router.route(task)
        assert result.delegated is False
        assert result.error == "No eligible peers available"
        assert result.peer is None


class TestSwarmRouterRoutingTable:
    def test_empty_table(self) -> None:
        router = SwarmRouter()
        assert router.routing_table() == []

    def test_sorted_by_routing_value(self) -> None:
        router = SwarmRouter()
        router.register_peer(SwarmPeer("low", "http://low:80", asset=0.2, risk=0.5))
        router.register_peer(SwarmPeer("high", "http://high:80", asset=1.0, risk=0.1))
        table = router.routing_table()
        assert table[0]["org_id"] == "high"


class TestSwarmRouterTaskPayload:
    def test_task_to_payload_serialisable(self) -> None:
        task = BrainTask(prompt="hello", domain="finance", stakes=0.8)
        payload = SwarmRouter._task_to_payload(task)
        parsed = json.loads(payload)
        assert parsed["prompt"] == "hello"
        assert parsed["domain"] == "finance"
        assert parsed["stakes"] == pytest.approx(0.8)

    def test_all_fields_present(self) -> None:
        task = BrainTask(prompt="test", domain="general")
        payload = SwarmRouter._task_to_payload(task)
        parsed = json.loads(payload)
        for key in ("prompt", "domain", "stakes", "uncertainty", "complexity"):
            assert key in parsed


# ---------------------------------------------------------------------------
# Integration: SwarmRouter with a live mock HTTP server
# ---------------------------------------------------------------------------


class _MockShieldHandler(BaseHTTPRequestHandler):
    """Minimal handler that returns a fixed /shield response."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        self.rfile.read(length)
        body = json.dumps({"vetoed": False, "risk_score": 0.1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def _start_mock_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), _MockShieldHandler)
    port: int = server.server_address[1]  # type: ignore[index]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()
    return server, port


class TestSwarmRouterHTTPDelegation:
    def test_successful_delegation(self) -> None:
        server, port = _start_mock_server()
        try:
            router = SwarmRouter(timeout=3.0)
            peer = SwarmPeer("mock-peer", f"http://127.0.0.1:{port}")
            router.register_peer(peer)
            task = BrainTask(prompt="delegate me", domain="general")
            result = router.route(task)
            assert result.delegated is True
            assert result.peer is not None
            assert result.peer.org_id == "mock-peer"
            assert result.response is not None
            assert result.response.get("vetoed") is False
        finally:
            server.server_close()

    def test_failed_delegation_bad_endpoint(self) -> None:
        router = SwarmRouter(timeout=1.0)
        # Use a port that is definitely not listening
        peer = SwarmPeer("dead-peer", "http://127.0.0.1:19999")
        router.register_peer(peer)
        task = BrainTask(prompt="test", domain="general")
        result = router.route(task)
        assert result.delegated is False
        assert result.error != ""
