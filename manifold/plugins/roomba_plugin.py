"""manifold/plugins/roomba_plugin.py — Roomba plugin for MANIFOLD.

Concrete ManifoldPlugin implementation with the Roomba iRobot profile
pre-filled.  Subclass or instantiate directly.

Usage::

    from manifold.plugins.roomba_plugin import RoombaPlugin
    from manifold.plugin import PluginRegistry

    registry = PluginRegistry()
    registry.register(RoombaPlugin(robot_id="roomba-kitchen"))
    registry.start_all()
"""

from __future__ import annotations

from manifold.plugin import ManifoldPlugin


class RoombaPlugin(ManifoldPlugin):
    """MANIFOLD plugin pre-configured for iRobot Roomba vacuum robots.

    Parameters
    ----------
    robot_id:
        Unique identifier for this Roomba.
    display_name:
        Human-readable name (defaults to ``"Roomba <robot_id>"``).
    """

    def __init__(
        self,
        robot_id: str = "roomba-01",
        display_name: str = "",
    ) -> None:
        self._robot_id = robot_id
        self._display_name = display_name or f"Roomba {robot_id}"
        self._status = "active"
        self._battery: float = 100.0
        self._position: dict | None = {"x": 0, "y": 0, "z": 0}

    # ------------------------------------------------------------------
    # ManifoldPlugin interface
    # ------------------------------------------------------------------

    def manifest(self) -> dict:
        return {
            "agent_id": self._robot_id,
            "display_name": self._display_name,
            "version": "1.0.0",
            "capabilities": ["vacuum", "map", "floor_nav"],
            "domain": "physical/kitchen",
            "layer": "physical/floor",
            "crna_profile": {"c": 0.3, "r": 0.3, "n": 0.2, "a": 0.0},
            "input_schema": ["start", "stop", "pause", "resume", "redirect", "dock"],
            "output_schema": ["obstacle", "bump", "clean_complete", "battery_low"],
        }

    def on_command(self, command: str, payload: dict) -> bool:
        if command == "pause":
            self._status = "paused"
            return True
        if command in ("resume", "start"):
            self._status = "active"
            return True
        if command in ("stop", "dock"):
            self._status = "idle"
            return True
        if command == "redirect":
            target = payload.get("target")
            if target:
                self._position = target
            return True
        return False

    def get_state(self) -> dict:
        return {
            "position": self._position,
            "health": 1.0 if self._status != "error" else 0.0,
            "status": self._status,
            "battery": self._battery,
        }

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_connect(self) -> bool:
        self._status = "active"
        return True

    def on_disconnect(self) -> None:
        self._status = "offline"
