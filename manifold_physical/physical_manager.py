"""PhysicalManager — unified manager for all MANIFOLD physical bridges.

Initialises RoombaBridge, MQTTBridge, and CameraDetector from a config dict
and provides a single start/stop/status interface for the physical layer.

Config dict format::

    {
      "roomba": {
        "robot_id": "roomba-01",
        "cloud_email": "you@example.com",
        "cloud_password": "secret",
        "grid_position": [0, 0, 0],
        "mock_mode": false
      },
      "mqtt": {
        "broker_host": "homeassistant.local",
        "broker_port": 1883,
        "devices": [
          {"topic": "homeassistant/binary_sensor/+/motion/state",
           "device_type": "motion_sensor",
           "grid_coord": [2, 3, 0],
           "risk_on_trigger": 0.75}
        ]
      },
      "cameras": [
        {"camera_index": 0, "model_size": "nano",
         "grid_origin": [0, 0, 0], "grid_scale": 1.0,
         "confidence_threshold": 0.45}
      ]
    }
"""

from __future__ import annotations

import logging
import time
from typing import Any

from manifold_physical.bridges.roomba_bridge import RoombaBridge
from manifold_physical.bridges.mqtt_bridge import DeviceMapping, MQTTBridge
from manifold_physical.camera_detector import CameraDetector


class PhysicalManager:
    """Manages all physical bridges in the MANIFOLD Physical layer.

    Parameters
    ----------
    config:
        Configuration dict (see module docstring for schema).
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._roomba: RoombaBridge | None = None
        self._mqtt: MQTTBridge | None = None
        self._cameras: list[CameraDetector] = []
        self._roomba_connected: bool = False
        self._mqtt_connected: bool = False
        self._last_obstacle_event: float | None = None
        self._agents_registered: int = 0

        if config:
            self._initialise(config)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self, config: dict) -> None:
        """Build bridges from the config dict without starting them."""
        # Roomba
        if "roomba" in config:
            rc = config["roomba"]
            self._roomba = RoombaBridge(
                robot_id=rc.get("robot_id", "roomba-01"),
                cloud_email=rc.get("cloud_email", ""),
                cloud_password=rc.get("cloud_password", ""),
                grid_position=tuple(rc.get("grid_position", [0, 0, 0])),
                manifold_url=rc.get("manifold_url", "http://localhost:8080"),
                manifold_api_key=rc.get("manifold_api_key", ""),
                mock_mode=rc.get("mock_mode", False),
            )

        # MQTT
        if "mqtt" in config:
            mc = config["mqtt"]
            self._mqtt = MQTTBridge(
                broker_host=mc.get("broker_host", "localhost"),
                broker_port=int(mc.get("broker_port", 1883)),
            )
            for dev in mc.get("devices", []):
                mapping = DeviceMapping(
                    topic=dev["topic"],
                    device_type=dev["device_type"],
                    grid_coord=tuple(dev.get("grid_coord", [0, 0, 0])),
                    risk_on_trigger=float(dev.get("risk_on_trigger", 0.75)),
                )
                self._mqtt.subscribe(dev["topic"], mapping)

        # Cameras
        for cam_cfg in config.get("cameras", []):
            det = CameraDetector(
                camera_index=int(cam_cfg.get("camera_index", 0)),
                model_size=cam_cfg.get("model_size", "nano"),
                grid_origin=tuple(cam_cfg.get("grid_origin", [0, 0, 0])),
                grid_scale=float(cam_cfg.get("grid_scale", 1.0)),
                confidence_threshold=float(cam_cfg.get("confidence_threshold", 0.45)),
            )
            self._cameras.append(det)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        """Connect and start all configured bridges."""
        if self._roomba is not None:
            self._roomba_connected = self._roomba.connect()
            if self._roomba_connected:
                self._roomba.start()
                self._agents_registered += 1
                logging.debug("PhysicalManager: Roomba started")

        if self._mqtt is not None:
            self._mqtt.start()
            self._mqtt_connected = self._mqtt.is_connected()
            if self._mqtt_connected:
                logging.debug("PhysicalManager: MQTT bridge started")

        for cam in self._cameras:
            cam.start()
            logging.debug("PhysicalManager: camera %s started", cam.camera_index)

    def stop_all(self) -> None:
        """Stop all bridges and release resources."""
        if self._roomba is not None:
            self._roomba.stop()
        if self._mqtt is not None:
            self._mqtt.disconnect()
        for cam in self._cameras:
            cam.stop()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return a status summary of the physical layer."""
        return {
            "roomba_connected": self._roomba_connected,
            "mqtt_connected": self._mqtt_connected,
            "cameras_running": sum(1 for c in self._cameras if c.is_running()),
            "agents_registered": self._agents_registered,
            "last_obstacle_event": self._last_obstacle_event,
        }
