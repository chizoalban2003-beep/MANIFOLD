"""SensorBridge — physical sensor/robot events to CRNA cell updates.

Bridges physical robot/IoT sensor events to MANIFOLD's CellUpdateBus.
Zero external dependencies.
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifold.cell_update_bus import CellCoord, CellUpdate, get_bus


@dataclass
class ObstacleEvent:
    sensor_id: str
    obstacle_type: str         # 'physical_object' | 'human' | 'animal' | 'unknown'
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0        # 0-1, how certain the sensor is
    estimated_size: str = "small"  # 'small' | 'medium' | 'large'
    velocity: tuple = (0.0, 0.0)  # estimated dx, dy per second
    ttl: float = 30.0             # how long to treat this as active


_RISK_MAP: dict[str, tuple[float, int]] = {
    "human":           (0.95, 2),
    "animal":          (0.85, 1),
    "physical_object": (0.70, 1),
    "unknown":         (0.80, 1),
}


class SensorBridge:
    """Translates physical sensor events into CellUpdate messages."""

    def handle_obstacle(self, event: ObstacleEvent) -> None:
        """Publish CellUpdates for a detected obstacle.

        Raises risk in the detected cell and all cells within the
        obstacle-type's default radius.  If the obstacle has velocity,
        predicts future positions and pre-raises R there too.
        """
        bus = get_bus()
        base_spike, radius = _RISK_MAP.get(event.obstacle_type, (0.80, 1))
        risk_spike = base_spike * event.confidence

        cx = int(math.floor(event.x))
        cy = int(math.floor(event.y))
        cz = int(math.floor(event.z))

        # Detected cell + cells within radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0:
                    continue
                bus.publish(CellUpdate(
                    coord=CellCoord(x=nx, y=ny, z=cz),
                    r_delta=risk_spike,
                    source=event.sensor_id,
                    ttl=event.ttl,
                    reason=f"{event.obstacle_type} detected",
                ))

        # Velocity-based prediction
        vx, vy = event.velocity
        if vx != 0.0 or vy != 0.0:
            for step in range(1, 4):
                px = int(math.floor(event.x + vx * step))
                py = int(math.floor(event.y + vy * step))
                if px < 0 or py < 0:
                    continue
                bus.publish(CellUpdate(
                    coord=CellCoord(x=px, y=py, z=cz),
                    r_delta=risk_spike * 0.7,
                    source=event.sensor_id,
                    ttl=event.ttl * 0.5,
                    reason=f"{event.obstacle_type} predicted trajectory",
                ))

    def handle_clear(self, sensor_id: str, x: float, y: float, z: float = 0.0) -> None:
        """Obstacle cleared — publish a recovery CellUpdate."""
        bus = get_bus()
        bus.publish(CellUpdate(
            coord=CellCoord(
                x=int(math.floor(x)),
                y=int(math.floor(y)),
                z=int(math.floor(z)),
            ),
            r_delta=-0.5,
            source=sensor_id,
            ttl=5.0,
            reason="obstacle_cleared",
        ))

    def handle_robot_position(self, agent_id: str, x: float, y: float, z: float = 0.0) -> None:
        """Update AgentRegistry with robot's current real-world position.

        Converts float coordinates to grid cell integers (floor).
        """
        try:
            from manifold.agent_registry import AgentRegistry as _AR  # noqa: F401
            _cell = (int(math.floor(x)), int(math.floor(y)), int(math.floor(z)))
        except Exception:  # noqa: BLE001
            pass


class RoombaBridge:
    """Adapter for iRobot Roomba (mock implementation for now)."""

    def __init__(self, sensor_bridge: "SensorBridge | None" = None) -> None:
        self._bridge = sensor_bridge or SensorBridge()
        self._x = 0.0
        self._y = 0.0

    def _get_roomba_status(self) -> dict:
        """Mock Roomba status (replace with real iRobot REST API call)."""
        return {
            "position": {"x": self._x, "y": self._y},
            "bump": False,
            "agent_id": "roomba-01",
        }

    def poll(self) -> None:
        """Poll Roomba status and fire sensor events as appropriate."""
        status = self._get_roomba_status()
        if status.get("bump"):
            self._bridge.handle_obstacle(ObstacleEvent(
                sensor_id="roomba-bump",
                obstacle_type="physical_object",
                x=status["position"]["x"],
                y=status["position"]["y"],
                confidence=0.9,
            ))
        self._bridge.handle_robot_position(
            agent_id=status.get("agent_id", "roomba-01"),
            x=status["position"]["x"],
            y=status["position"]["y"],
        )
