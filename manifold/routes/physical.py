"""manifold/routes/physical.py — Physical layer endpoint handlers.

Handlers for:
  GET  /physical/cameras
  GET  /physical/status
  POST /physical/init
  GET  /nervatura/world
  POST /nervatura/world/init
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifold.server import ManifoldHandler


def _srv():
    import manifold.server as _s  # noqa: PLC0415
    return _s


def handle_get_physical_cameras(self: "ManifoldHandler") -> None:
    """GET /physical/cameras — list all registered camera detectors."""
    s = _srv()
    try:
        from manifold_physical.camera_detector import get_camera_registry  # noqa: PLC0415
        registry = get_camera_registry()
        s._send_json(self, 200, {"cameras": registry.status_list()})
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})


def handle_get_physical_status(self: "ManifoldHandler") -> None:
    """GET /physical/status — PhysicalManager status, or empty if not initialised."""
    s = _srv()
    with s._PHYSICAL_MANAGER_LOCK:
        pm = s._PHYSICAL_MANAGER
    if pm is None:
        s._send_json(self, 200, {
            "roomba_connected": False,
            "mqtt_connected": False,
            "cameras_running": 0,
            "agents_registered": 0,
            "last_obstacle_event": None,
            "initialised": False,
        })
        return
    try:
        status = pm.status()
        status["initialised"] = True
        s._send_json(self, 200, status)
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})


def handle_post_physical_init(self: "ManifoldHandler", body: dict) -> None:
    """POST /physical/init — initialise (or re-initialise) the PhysicalManager."""
    s = _srv()
    with s._PHYSICAL_MANAGER_LOCK:
        try:
            if s._PHYSICAL_MANAGER is not None:
                try:
                    s._PHYSICAL_MANAGER.stop_all()
                except Exception:  # noqa: BLE001
                    pass
            from manifold_physical.physical_manager import PhysicalManager  # noqa: PLC0415
            s._PHYSICAL_MANAGER = PhysicalManager(config=body)
            s._PHYSICAL_MANAGER.start_all()
            status = s._PHYSICAL_MANAGER.status()
        except Exception as exc:  # noqa: BLE001
            s._send_json(self, 500, {"error": str(exc)})
            return
    s._send_json(self, 200, {"status": "ok", **status})


def handle_get_nervatura_world(self: "ManifoldHandler") -> None:
    """GET /nervatura/world — NERVATURAWorld summary."""
    s = _srv()
    try:
        if s._NERVATURA is None:
            s._send_json(self, 200, {
                "status": "not_initialised",
                "hint": "POST /nervatura/world/init to create a world",
            })
        else:
            s._send_json(self, 200, s._NERVATURA.summary())
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})


def handle_post_nervatura_world_init(self: "ManifoldHandler", body: dict) -> None:
    """POST /nervatura/world/init — initialise NERVATURAWorld singleton."""
    s = _srv()
    try:
        width = int(body.get("width", 20))
        depth = int(body.get("depth", 20))
        height = int(body.get("height", 5))
        domain = str(body.get("domain", "general"))
        s._NERVATURA = s._NERVATURAWorld(width=width, depth=depth, height=height)
        # PROMPT 6: start ConvergenceMonitor now that world is initialised
        try:
            from manifold.convergence_monitor import ConvergenceMonitor  # noqa: PLC0415
            s._CONVERGENCE_MONITOR = ConvergenceMonitor(s._NERVATURA, window=50)
            s._CONVERGENCE_MONITOR.start(interval_seconds=30.0)
        except Exception:  # noqa: BLE001
            pass
        s._send_json(self, 200, {
            "status": "ok",
            "domain": domain,
            **s._NERVATURA.summary(),
        })
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})
