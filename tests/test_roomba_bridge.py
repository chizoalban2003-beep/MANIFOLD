"""Tests for manifold_physical/bridges/roomba_bridge.py — mock_mode=True."""

import time

import pytest

from manifold.cell_update_bus import CellUpdateBus
from manifold_physical.bridges.roomba_bridge import RoombaBridge
import manifold_physical.sensor_bridge as sb_mod


@pytest.fixture()
def mock_bus(monkeypatch):
    """Patch get_bus to return a fresh test bus."""
    bus = CellUpdateBus()
    monkeypatch.setattr(sb_mod, "get_bus", lambda: bus)
    return bus


def test_connect_mock_mode_returns_true():
    """connect() in mock_mode should succeed without network calls."""
    bridge = RoombaBridge(
        robot_id="roomba-test",
        cloud_email="test@example.com",
        cloud_password="secret",
        mock_mode=True,
    )
    result = bridge.connect()
    assert result is True
    assert bridge._cloud_token == "mock-token"


def test_mock_state_produces_valid_dict():
    """_mock_state() should return a dict with x, y, and bump_left / bump_right."""
    bridge = RoombaBridge(
        robot_id="roomba-state",
        cloud_email="test@example.com",
        cloud_password="secret",
        grid_position=(3, 4, 0),
        mock_mode=True,
    )
    state = bridge._mock_state()
    assert isinstance(state["x"], float)
    assert isinstance(state["y"], float)
    assert "bump_left" in state
    assert "bump_right" in state


def test_bump_sensor_triggers_obstacle_event(mock_bus):
    """Polling sensors when bump_left is True should publish a CellUpdate."""
    bridge = RoombaBridge(
        robot_id="roomba-bump",
        cloud_email="test@example.com",
        cloud_password="secret",
        mock_mode=True,
    )
    received = []
    mock_bus.subscribe("bump-test", lambda u: received.append(u))

    # Force bump_left to be True: mock triggers on count % 7 == 0
    bridge._mock_poll_count = 6  # next poll: count becomes 7, so bump_left = True
    bridge._poll_sensors()
    time.sleep(0.1)

    assert len(received) > 0
    assert any(u.r_delta > 0 for u in received)


def test_start_stop_does_not_raise():
    """start() and stop() in mock_mode should not raise any exceptions."""
    bridge = RoombaBridge(
        robot_id="roomba-start",
        cloud_email="test@example.com",
        cloud_password="secret",
        mock_mode=True,
    )
    bridge.start()
    time.sleep(0.05)
    bridge.stop()
    assert not bridge._running
