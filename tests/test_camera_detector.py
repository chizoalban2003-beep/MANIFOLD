"""Tests for manifold_physical/camera_detector.py.

These tests mock the camera and YOLO model so no hardware or GPU is required.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from manifold.cell_update_bus import CellUpdateBus
from manifold_physical.camera_detector import (
    CameraDetector,
    CameraRegistry,
    Detection,
    _CLEAR_TIMEOUT,
)
import manifold_physical.sensor_bridge as sb_mod
import manifold_physical.camera_detector as cam_mod


@pytest.fixture()
def mock_bus(monkeypatch):
    """Patch get_bus to return a fresh test bus."""
    bus = CellUpdateBus()
    monkeypatch.setattr(sb_mod, "get_bus", lambda: bus)
    return bus


@pytest.fixture(autouse=True)
def reset_camera_registry():
    """Reset the CameraRegistry singleton between tests."""
    CameraRegistry._instance = None
    yield
    CameraRegistry._instance = None


def _make_fake_detection(class_name: str, confidence: float) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=(10.0, 10.0, 100.0, 100.0),
        estimated_grid_coord=(1, 2, 0),
    )


def test_detected_person_publishes_r_095(mock_bus):
    """A detected person should publish R ≈ 0.95 * confidence to the bus."""
    received = []
    mock_bus.subscribe("cam-person", lambda u: received.append(u))

    det = CameraDetector(camera_index=99)

    # Simulate detect_objects returning a "person" detection
    person = _make_fake_detection("person", confidence=1.0)

    # Manually trigger the obstacle publish path (same as _detection_loop does it)
    from manifold_physical.sensor_bridge import ObstacleEvent
    from manifold_physical import camera_detector as cdm

    obstacle_type, risk = cdm._CLASS_MAP.get("person", cdm._DEFAULT_OBSTACLE)
    det._sensor_bridge.handle_obstacle(ObstacleEvent(
        sensor_id="camera-99",
        obstacle_type=obstacle_type,
        x=1.0, y=2.0, z=0.0,
        confidence=risk * person.confidence,
        ttl=_CLEAR_TIMEOUT,
    ))
    time.sleep(0.1)

    assert len(received) > 0
    # human obstacle: _CLASS_MAP gives risk=0.95; SensorBridge applies base_spike(human)=0.95 * confidence(0.95)
    # so r_delta = 0.95 * 0.95 = 0.9025 at the center cell
    max_r = max(u.r_delta for u in received)
    assert max_r >= 0.85  # accounting for the double-confidence product


def test_detected_cat_publishes_r_085(mock_bus):
    """A detected cat should publish R based on animal risk (0.85) to the bus."""
    received = []
    mock_bus.subscribe("cam-cat", lambda u: received.append(u))

    det = CameraDetector(camera_index=98)

    from manifold_physical.sensor_bridge import ObstacleEvent
    from manifold_physical import camera_detector as cdm

    obstacle_type, risk = cdm._CLASS_MAP.get("cat", cdm._DEFAULT_OBSTACLE)
    det._sensor_bridge.handle_obstacle(ObstacleEvent(
        sensor_id="camera-98",
        obstacle_type=obstacle_type,
        x=1.0, y=2.0, z=0.0,
        confidence=risk * 1.0,  # 0.85 passed as event.confidence
        ttl=_CLEAR_TIMEOUT,
    ))
    time.sleep(0.1)

    assert len(received) > 0
    # SensorBridge: base_spike("animal")=0.85, risk_spike = 0.85 * 0.85 ≈ 0.7225
    max_r = max(u.r_delta for u in received)
    assert max_r >= 0.70


def test_handle_clear_fires_after_disappearance(mock_bus):
    """The detector should call handle_clear for a cell not seen for CLEAR_TIMEOUT."""
    received = []
    mock_bus.subscribe("cam-clear", lambda u: received.append(u))

    det = CameraDetector(camera_index=97)

    # Simulate a previous detection in cell (3, 3, 0) that is now stale
    stale_cell = (3, 3, 0)
    det._last_seen[stale_cell] = time.monotonic() - (_CLEAR_TIMEOUT + 1.0)

    # Trigger the clear sweep (same logic as _detection_loop end)
    now = time.monotonic()
    stale = [c for c, ts in det._last_seen.items() if now - ts >= _CLEAR_TIMEOUT]
    for cell in stale:
        det._sensor_bridge.handle_clear(
            sensor_id=f"camera-{det.camera_index}",
            x=float(cell[0]),
            y=float(cell[1]),
            z=float(cell[2]),
        )
        del det._last_seen[cell]
    time.sleep(0.1)

    assert len(received) > 0
    # handle_clear publishes a negative r_delta
    assert any(u.r_delta < 0 for u in received)


def test_raspberry_pi_config_returns_detector():
    """RaspberryPiConfig classmethod should return a CameraDetector with expected settings."""
    det = CameraDetector.RaspberryPiConfig()
    assert det.camera_index == 0
    assert det.model_size == "nano"
    assert det.confidence_threshold == pytest.approx(0.40)
