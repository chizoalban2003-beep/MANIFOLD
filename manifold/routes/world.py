"""manifold/routes/world.py — MANIFOLD World and real-time status handlers.

Handlers for:
  GET /world
  GET /world/manifest.json
  GET /ws  (WebSocket upgrade)
  GET /health/tools
  GET /realtime/status
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.server import ManifoldHandler


def _srv():
    import manifold.server as _s  # noqa: PLC0415
    return _s


def handle_get_world(self: "ManifoldHandler") -> None:
    """GET /world — serve the isometric game world HTML."""
    import os  # noqa: PLC0415
    s = _srv()
    world_file = os.path.join(s._WORLD_DIR, "index.html")
    if not os.path.exists(world_file):
        s._send_error(self, 404, "MANIFOLD World not found")
        return
    with open(world_file, "rb") as fh:
        data = fh.read()
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def handle_get_world_manifest(self: "ManifoldHandler") -> None:
    """GET /world/manifest.json — serve the PWA manifest."""
    import os  # noqa: PLC0415
    s = _srv()
    manifest_file = os.path.join(s._WORLD_DIR, "manifest.json")
    if not os.path.exists(manifest_file):
        s._send_error(self, 404, "Manifest not found")
        return
    with open(manifest_file, "rb") as fh:
        data = fh.read()
    self.send_response(200)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def handle_ws_upgrade(self: "ManifoldHandler") -> None:
    """GET /ws — WebSocket upgrade + live event loop."""
    import base64  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    s = _srv()
    upgrade = self.headers.get("Upgrade", "").lower()
    if upgrade != "websocket":
        s._send_error(self, 400, "WebSocket upgrade required")
        return
    key = self.headers.get("Sec-WebSocket-Key", "")
    if not key:
        s._send_error(self, 400, "Sec-WebSocket-Key missing")
        return
    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = base64.b64encode(
        hashlib.sha1((key + magic).encode()).digest()
    ).decode()
    self.send_response(101)
    self.send_header("Upgrade", "websocket")
    self.send_header("Connection", "Upgrade")
    self.send_header("Sec-WebSocket-Accept", accept)
    self.end_headers()
    self.wfile.flush()

    conn = self.connection
    conn.settimeout(1.0)

    last_agent_push = 0.0
    last_stats_push = 0.0

    def _agents_payload() -> bytes:
        agents = s._AGENT_REGISTRY.all_agents()
        data = {
            "type": "agent_update",
            "agents": [
                {
                    "id": a.agent_id,
                    "status": a.status,
                    "task": a.notes or "active",
                    "risk": round(1.0 - a.health_score(), 4),
                    "health": round(a.health_score(), 4),
                    "tx": float(a.task_count % 16),
                    "tz": float(a.error_count % 16),
                }
                for a in agents
            ],
        }
        return json.dumps(data).encode()

    def _stats_payload() -> bytes:
        summary = s._AGENT_REGISTRY.summary()
        plans = s._TASK_ROUTER.all_plans()
        data = {
            "type": "world_stats",
            "agents_active": summary.get("active", 0),
            "tasks_running": sum(1 for p in plans if not p.executable and p.blocked_count == 0),
            "governance_events_today": s._REFUSAL_COUNT,
            "system_health": round(summary.get("avg_health", 1.0), 4),
        }
        return json.dumps(data).encode()

    try:
        while True:
            now = time.time()
            if now - last_agent_push >= 5.0:
                s._ws_send_frame(conn, _agents_payload())
                last_agent_push = now
            if now - last_stats_push >= 30.0:
                s._ws_send_frame(conn, _stats_payload())
                last_stats_push = now
            try:
                frame = s._ws_read_frame(conn)
                if frame is None:
                    break
            except OSError:
                pass
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def handle_get_health_tools(self: "ManifoldHandler") -> None:
    """GET /health/tools — live tool health summary (public)."""
    s = _srv()
    try:
        s._send_json(self, 200, s._HEALTH_MONITOR.status())
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})


def handle_get_realtime_status(self: "ManifoldHandler") -> None:
    """GET /realtime/status — live bus, grid, health, planner status."""
    s = _srv()
    try:
        bus = s._get_bus()
        convergence_warning = None
        if s._CONVERGENCE_MONITOR is not None:
            report = s._CONVERGENCE_MONITOR.convergence_report()
            if report.get("health") == "diverging":
                convergence_warning = report.get("recommendation", "NERVATURA diverging")
        s._send_json(self, 200, {
            "bus_recent_updates": len(bus.recent()),
            "dynamic_grid_cells": len(s._DYNAMIC_GRID.all_cells()),
            "health_monitor": s._HEALTH_MONITOR.status(),
            "planner_ready": True,
            "nervatura_world": s._NERVATURA.summary() if s._NERVATURA is not None else None,
            "convergence_warning": convergence_warning,
        })
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})
