# manifold_physical — Python-importable package for MANIFOLD Physical.
# The human-readable directory is manifold-physical/ (with hyphen);
# this package (manifold_physical, with underscore) is the importable alias.
from manifold_physical.space_ingestion import SpaceIngestion
from manifold_physical.sensor_bridge import SensorBridge, ObstacleEvent, RoombaBridge
from manifold_physical.bridges.roomba_bridge import RoombaBridge as RoombaBridgeFull
from manifold_physical.bridges.mqtt_bridge import MQTTBridge, DeviceMapping
from manifold_physical.camera_detector import (
    CameraDetector,
    CameraRegistry,
    Detection,
    get_camera_registry,
)
from manifold_physical.physical_manager import PhysicalManager

__all__ = [
    "SpaceIngestion",
    "SensorBridge",
    "ObstacleEvent",
    "RoombaBridge",
    "RoombaBridgeFull",
    "MQTTBridge",
    "DeviceMapping",
    "CameraDetector",
    "CameraRegistry",
    "Detection",
    "get_camera_registry",
    "PhysicalManager",
]
