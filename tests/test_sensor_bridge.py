"""Tests for manifold_physical/sensor_bridge.py."""
import time

import pytest

from manifold.cell_update_bus import CellUpdateBus
from manifold_physical.sensor_bridge import ObstacleEvent, RoombaBridge, SensorBridge


@pytest.fixture()
def bridge_and_bus(monkeypatch):
    """SensorBridge wired to a fresh test bus."""
    bus = CellUpdateBus()
    import manifold_physical.sensor_bridge as sb_mod
    monkeypatch.setattr(sb_mod, "get_bus", lambda: bus)
    bridge = SensorBridge()
    yield bridge, bus


def test_handle_obstacle_fires_cellupdate(bridge_and_bus):
    bridge, bus = bridge_and_bus
    received = []
    bus.subscribe("test-obs", lambda u: received.append(u))
    bridge.handle_obstacle(ObstacleEvent(
        sensor_id="lidar-01",
        obstacle_type="physical_object",
        x=3.0, y=3.0,
        confidence=1.0,
    ))
    time.sleep(0.1)
    assert len(received) > 0


def test_human_obstacle_r_spike_at_least_095_times_confidence(bridge_and_bus):
    bridge, bus = bridge_and_bus
    received = []
    bus.subscribe("test-human", lambda u: received.append(u))
    bridge.handle_obstacle(ObstacleEvent(
        sensor_id="cam-01",
        obstacle_type="human",
        x=5.0, y=5.0,
        confidence=1.0,
    ))
    time.sleep(0.1)
    # At the detected cell (5,5,0) the r_delta must be >= 0.95
    center_updates = [u for u in received if u.coord.x == 5 and u.coord.y == 5]
    assert len(center_updates) > 0
    assert any(u.r_delta >= 0.95 for u in center_updates)


def test_obstacle_with_velocity_pre_raises_r_in_predicted_cells(bridge_and_bus):
    bridge, bus = bridge_and_bus
    received = []
    bus.subscribe("test-vel", lambda u: received.append(u))
    bridge.handle_obstacle(ObstacleEvent(
        sensor_id="lidar-02",
        obstacle_type="physical_object",
        x=0.0, y=0.0,
        velocity=(1.0, 0.0),  # moving in +x direction
        ttl=30.0,
    ))
    time.sleep(0.1)
    # Should see predicted cells at (1,0), (2,0), (3,0)
    predicted = [u for u in received if u.coord.x > 0 and u.coord.y == 0]
    assert len(predicted) > 0


def test_handle_clear_publishes_negative_r_delta(bridge_and_bus):
    bridge, bus = bridge_and_bus
    received = []
    bus.subscribe("test-clear", lambda u: received.append(u))
    bridge.handle_clear("lidar-01", x=3.0, y=3.0)
    time.sleep(0.1)
    assert len(received) == 1
    assert received[0].r_delta < 0


def test_roomba_bridge_instantiates_without_error():
    rb = RoombaBridge()
    assert rb is not None
    # poll() should not raise
    rb.poll()
