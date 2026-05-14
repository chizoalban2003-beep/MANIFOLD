"""manifold/routes/federation.py — Federation and ATS endpoint handlers.

Handlers for:
  GET  /federation/status
  POST /federation/join
  POST /federation/gossip
  POST /ats/register
  POST /ats/signal
  GET  /ats/leaderboard
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manifold.server import ManifoldHandler


def _srv():
    import manifold.server as _s  # noqa: PLC0415
    return _s


def handle_get_federation_status(self: "ManifoldHandler") -> None:
    """GET /federation/status — return federation health summary."""
    s = _srv()
    ledger = s._GOSSIP_BRIDGE.ledger
    known = ledger.known_tools()
    trust_entries = {t: ledger.global_rate(t) for t in known}
    contributing_orgs = len(s._GOSSIP_BRIDGE.registered_orgs())
    s._send_json(
        self,
        200,
        {
            "contributing_orgs": contributing_orgs,
            "known_tools": len(known),
            "trust_entries": trust_entries,
        },
    )


def handle_post_federation_join(
    self: "ManifoldHandler", body: dict, caller: Any
) -> None:
    """POST /federation/join — register calling org with the gossip bridge."""
    s = _srv()
    org_id = caller.org_id if caller else body.get("org_id", "")
    if not org_id:
        s._send_error(self, 400, "org_id required")
        return
    s._GOSSIP_BRIDGE.register(org_id)
    s._send_json(
        self,
        200,
        {
            "org_id": org_id,
            "status": "joined",
            "contributing_orgs": len(s._GOSSIP_BRIDGE.registered_orgs()),
        },
    )


def handle_post_federation_gossip(self: "ManifoldHandler", body: dict) -> None:
    """POST /federation/gossip — ingest a gossip packet."""
    from manifold.federation import FederatedGossipPacket  # noqa: PLC0415
    s = _srv()
    try:
        packet = FederatedGossipPacket(**body)
        s._GOSSIP_BRIDGE.contribute_packet(packet)
        s._send_json(self, 200, {"status": "ingested"})
    except Exception as exc:  # noqa: BLE001
        s._send_error(self, 400, f"Invalid gossip packet: {exc}")


def handle_get_federation_bft_status(self: "ManifoldHandler") -> None:
    """GET /federation/bft-status — return BFT quorum status."""
    s = _srv()
    bridge = s._GOSSIP_BRIDGE
    node_count = len(bridge.registered_orgs())
    s._send_json(
        self,
        200,
        {
            "bft_active": bridge.bft_enabled,
            "node_count": node_count,
            "quorum": bridge.quorum,
            "f": bridge.f,
        },
    )


def handle_post_ats_register(self: "ManifoldHandler", body: dict[str, Any]) -> None:
    """POST /ats/register — register a tool in the ATS network."""
    s = _srv()
    from manifold.trust_network.models import ToolRegistration  # noqa: PLC0415
    tool_id = body.get("tool_id", "")
    org_id = body.get("org_id", "")
    display_name = body.get("display_name", tool_id)
    domain = body.get("domain", "general")
    description = body.get("description", "")
    if not tool_id:
        s._send_error(self, 400, "tool_id required")
        return
    reg = ToolRegistration(
        tool_id=tool_id,
        org_id=org_id,
        display_name=display_name,
        domain=domain,
        description=description,
    )
    s._ats_registry.register(reg)
    s._send_json(self, 201, {"tool_id": tool_id, "status": "registered"})


def handle_post_ats_signal(self: "ManifoldHandler", body: dict[str, Any]) -> None:
    """POST /ats/signal — submit a trust signal for a tool."""
    s = _srv()
    from manifold.trust_network.models import TrustSignal  # noqa: PLC0415
    tool_id = body.get("tool_id", "")
    signal_type = body.get("signal_type", "success")
    domain = body.get("domain", "general")
    stakes = float(body.get("stakes", 0.5))
    submitter_hash = body.get("submitter_hash", "")
    metadata = body.get("metadata", {})
    if not tool_id:
        s._send_error(self, 400, "tool_id required")
        return
    sig = TrustSignal(
        tool_id=tool_id,
        signal_type=signal_type,
        domain=domain,
        stakes=stakes,
        submitter_hash=submitter_hash,
        metadata=metadata,
    )
    s._ats_registry.submit_signal(sig)
    s._send_json(self, 200, {"tool_id": tool_id, "status": "signal_recorded"})


def handle_get_ats_leaderboard(self: "ManifoldHandler") -> None:
    """GET /ats/leaderboard — return top 10 tools by trust score."""
    s = _srv()
    board = s._ats_registry.leaderboard(top_n=10)
    s._send_json(self, 200, [e.to_dict() for e in board])
