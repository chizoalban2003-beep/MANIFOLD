"""manifold/routes/agents.py — Agent lifecycle endpoint handlers.

Handlers for:
  GET  /agents
  POST /agents/register
  POST /agents/{id}/heartbeat
  POST /agents/{id}/pause
  POST /agents/{id}/resume
  GET  /agents/{id}/commands
  POST /agents/{id}/command
  GET  /grid/occupancy        (stub — served via realtime status)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.server import ManifoldHandler


# ---------------------------------------------------------------------------
# Helpers (lazy import pattern to avoid circular imports)
# ---------------------------------------------------------------------------

def _srv():
    import manifold.server as _s  # noqa: PLC0415
    return _s


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_get_agents(self: "ManifoldHandler") -> None:
    """GET /agents — list all registered agents."""
    s = _srv()
    s._send_json(self, 200, {
        "agents": [a.to_dict() for a in s._AGENT_REGISTRY.all_agents()],
        "summary": s._AGENT_REGISTRY.summary(),
        "monitor": s._AGENT_MONITOR.status(),
    })


def handle_post_agents_register(self: "ManifoldHandler", body: dict) -> None:
    """POST /agents/register — agent announces itself."""
    s = _srv()
    agent_id = str(body.get("agent_id", "")).strip()
    name = str(body.get("display_name", "")).strip()
    caps = body.get("capabilities", [])
    org_id = str(body.get("org_id", "default")).strip()
    endpoint = str(body.get("endpoint_url", "")).strip()
    domain = str(body.get("domain", "general")).strip()
    if not agent_id or not name:
        s._send_error(self, 400, "agent_id and display_name required")
        return
    record = s._AGENT_REGISTRY.register(
        agent_id=agent_id,
        display_name=name,
        capabilities=caps if isinstance(caps, list) else [],
        org_id=org_id,
        endpoint_url=endpoint,
        domain=domain,
    )
    s._send_json(self, 201, record.to_dict())


def handle_post_agent_heartbeat(
    self: "ManifoldHandler", agent_id: str, body: dict
) -> None:
    """POST /agents/{id}/heartbeat — keep-alive."""
    s = _srv()
    status = str(body.get("status", "active"))
    ok = s._AGENT_REGISTRY.heartbeat(agent_id, status)
    if not ok:
        s._send_error(self, 404, f"Agent {agent_id!r} not registered")
        return
    s._send_json(self, 200, {"agent_id": agent_id, "acknowledged": True})


def handle_post_agent_pause(self: "ManifoldHandler", agent_id: str) -> None:
    """POST /agents/{id}/pause — MANIFOLD pauses an agent."""
    s = _srv()
    ok = s._AGENT_REGISTRY.pause(agent_id)
    if not ok:
        s._send_error(self, 404, f"Agent {agent_id!r} not found")
        return
    s._send_json(self, 200, {"agent_id": agent_id, "status": "paused"})


def handle_post_agent_resume(self: "ManifoldHandler", agent_id: str) -> None:
    """POST /agents/{id}/resume — MANIFOLD resumes a paused agent."""
    s = _srv()
    ok = s._AGENT_REGISTRY.resume(agent_id)
    if not ok:
        s._send_error(self, 404, f"Agent {agent_id!r} not found")
        return
    s._send_json(self, 200, {"agent_id": agent_id, "status": "active"})


def handle_get_agent_commands(self: "ManifoldHandler", agent_id: str) -> None:
    """GET /agents/{id}/commands — long-poll for up to 20 seconds."""
    import time  # noqa: PLC0415
    s = _srv()
    deadline = time.time() + 20.0
    while time.time() < deadline:
        cmds = s._AGENT_REGISTRY.poll_commands(agent_id, consume=True)
        if cmds:
            s._send_json(self, 200, {"commands": cmds, "agent_id": agent_id})
            return
        time.sleep(0.5)
    s._send_json(self, 200, {"commands": [], "agent_id": agent_id})


def handle_post_agent_command(
    self: "ManifoldHandler", agent_id: str, body: dict
) -> None:
    """POST /agents/{id}/command — queue a command for an agent."""
    s = _srv()
    command = str(body.get("command", "")).strip()
    payload = body.get("payload", {})
    valid = {"pause", "resume", "redirect", "update_policy", "message"}
    if command not in valid:
        s._send_error(self, 400, f"Invalid command. Must be one of: {sorted(valid)}")
        return
    cmd_id = s._AGENT_REGISTRY.queue_command(agent_id, command, payload)
    if cmd_id is None:
        s._send_error(self, 404, f"Agent {agent_id!r} not registered")
        return
    s._send_json(
        self,
        201,
        {
            "command_id": cmd_id,
            "agent_id": agent_id,
            "command": command,
            "status": "queued",
        },
    )
