"""Tests for manifold_physical/bridges/mqtt_bridge.py using mock socket."""

import time

import pytest

from manifold.cell_update_bus import CellUpdateBus
from manifold_physical.bridges.mqtt_bridge import (
    DeviceMapping,
    MQTTBridge,
    _encode_remaining,
    _topic_matches,
)
import manifold_physical.sensor_bridge as sb_mod


@pytest.fixture()
def mock_bus(monkeypatch):
    """Patch get_bus to return a fresh test bus."""
    bus = CellUpdateBus()
    monkeypatch.setattr(sb_mod, "get_bus", lambda: bus)
    return bus


def test_device_mapping_defaults():
    """DeviceMapping should accept minimal arguments with sensible defaults."""
    dm = DeviceMapping(
        topic="home/sensor/motion",
        device_type="motion_sensor",
    )
    assert dm.grid_coord == (0, 0, 0)
    assert dm.risk_on_trigger == 0.75


def test_home_assistant_profile_returns_three_mappings():
    """HomeAssistantProfile classmethod should return 3 DeviceMapping objects."""
    profile = MQTTBridge.HomeAssistantProfile()
    assert len(profile) == 3
    topics = [m.topic for m in profile]
    assert any("motion" in t for t in topics)
    assert any("door" in t for t in topics)
    assert any("smoke" in t for t in topics)


def test_subscribe_registers_mapping():
    """subscribe() should add the mapping to the internal dict."""
    bridge = MQTTBridge(broker_host="localhost")
    mapping = DeviceMapping(
        topic="test/sensor/motion",
        device_type="motion_sensor",
        grid_coord=(1, 2, 0),
    )
    bridge.subscribe("test/sensor/motion", mapping)
    assert "test/sensor/motion" in bridge._mappings


def test_publish_motion_fires_obstacle(mock_bus):
    """Simulated PUBLISH with an 'ON' payload on a motion sensor should publish R > 0."""
    bridge = MQTTBridge(broker_host="localhost")
    mapping = DeviceMapping(
        topic="home/motion",
        device_type="motion_sensor",
        grid_coord=(3, 4, 0),
        risk_on_trigger=0.75,
    )
    bridge.subscribe("home/motion", mapping)

    received = []
    mock_bus.subscribe("mqtt-test-motion", lambda u: received.append(u))

    bridge._simulate_publish("home/motion", "ON")
    time.sleep(0.1)

    assert len(received) > 0
    assert any(u.r_delta > 0 for u in received)


def test_idle_payload_fires_handle_clear(mock_bus):
    """Simulated PUBLISH with an 'OFF' payload should publish a negative r_delta (clear)."""
    bridge = MQTTBridge(broker_host="localhost")
    mapping = DeviceMapping(
        topic="home/door",
        device_type="door_sensor",
        grid_coord=(5, 6, 0),
    )
    bridge.subscribe("home/door", mapping)

    received = []
    mock_bus.subscribe("mqtt-test-clear", lambda u: received.append(u))

    bridge._simulate_publish("home/door", "off")
    time.sleep(0.1)

    # handle_clear publishes a negative r_delta
    assert len(received) > 0
    assert any(u.r_delta < 0 for u in received)


def test_wildcard_topic_matching():
    """_topic_matches should handle both + and # wildcards correctly."""
    assert _topic_matches("home/+/motion", "home/living_room/motion")
    assert _topic_matches("home/#", "home/a/b/c")
    assert not _topic_matches("home/+/motion", "office/sensor/motion")
    assert not _topic_matches("home/motion", "home/sensor/motion")


def test_encode_remaining_single_byte():
    """_encode_remaining(0) should return b'\\x00'."""
    assert _encode_remaining(0) == b"\x00"
    assert _encode_remaining(127) == b"\x7f"


def test_smoke_detector_uses_high_risk(mock_bus):
    """Smoke detector should override risk to 0.95 regardless of mapping.risk_on_trigger."""
    bridge = MQTTBridge(broker_host="localhost")
    mapping = DeviceMapping(
        topic="home/smoke",
        device_type="smoke_detector",
        grid_coord=(1, 1, 0),
        risk_on_trigger=0.5,  # should be overridden to 0.95
    )
    bridge.subscribe("home/smoke", mapping)

    received = []
    mock_bus.subscribe("mqtt-smoke", lambda u: received.append(u))

    bridge._simulate_publish("home/smoke", "SMOKE_DETECTED")
    time.sleep(0.1)

    assert len(received) > 0
    # The risk override for smoke_detector is 0.95, so r_delta should be close
    max_r = max(u.r_delta for u in received)
    assert max_r > 0.5
