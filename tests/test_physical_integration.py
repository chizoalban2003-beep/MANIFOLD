"""Integration tests for MANIFOLD Physical v0.1 (P4).

Tests:
1. PhysicalManager initialises without real hardware using mock_mode.
2. PhysicalManager.status() returns the expected shape.
3. Obstacle from mock Roomba appears in DynamicGrid.
"""

import time

import pytest

from manifold_physical.physical_manager import PhysicalManager
from manifold_physical.bridges.roomba_bridge import RoombaBridge
from manifold.cell_update_bus import CellUpdateBus
import manifold_physical.sensor_bridge as sb_mod


@pytest.fixture()
def mock_bus(monkeypatch):
    bus = CellUpdateBus()
    monkeypatch.setattr(sb_mod, "get_bus", lambda: bus)
    return bus


# ---------------------------------------------------------------------------
# Test 1: PhysicalManager initialises without hardware
# ---------------------------------------------------------------------------

def test_physical_manager_initialises_with_mock_mode():
    """PhysicalManager should initialise Roomba in mock_mode without network calls."""
    config = {
        "roomba": {
            "robot_id": "roomba-integ",
            "cloud_email": "test@example.com",
            "cloud_password": "secret",
            "grid_position": [0, 0, 0],
            "mock_mode": True,
        }
    }
    pm = PhysicalManager(config=config)
    # Roomba should be created
    assert pm._roomba is not None
    assert pm._roomba.mock_mode is True
    # start_all should not raise
    pm.start_all()
    time.sleep(0.05)
    pm.stop_all()
    # After connect() in mock_mode, roomba_connected should be True
    assert pm._roomba_connected is True


# ---------------------------------------------------------------------------
# Test 2: GET /physical/status returns expected shape
# ---------------------------------------------------------------------------

def test_get_physical_status_returns_expected_shape():
    """PhysicalManager.status() should return a dict with expected keys."""
    pm = PhysicalManager(config={})
    data = pm.status()
    assert "roomba_connected" in data
    assert "mqtt_connected" in data
    assert "cameras_running" in data
    assert "agents_registered" in data
    assert "last_obstacle_event" in data


# ---------------------------------------------------------------------------
# Test 3: Obstacle from mock Roomba appears in DynamicGrid
# ---------------------------------------------------------------------------

def test_mock_roomba_obstacle_updates_dynamic_grid():
    """Polling sensors on a mock Roomba with bump_left=True should publish to the real bus."""
    from manifold.cell_update_bus import get_bus

    bus = get_bus()
    received = []
    bus.subscribe("test-roomba-grid", lambda u: received.append(u))

    try:
        bridge = RoombaBridge(
            robot_id="roomba-grid-test",
            cloud_email="test@example.com",
            cloud_password="secret",
            grid_position=(10, 10, 0),
            mock_mode=True,
        )

        # Force bump_left=True: _mock_state increments count to 7 → 7%7==0
        bridge._mock_poll_count = 6
        bridge._poll_sensors()
        time.sleep(0.3)
    finally:
        bus.unsubscribe("test-roomba-grid")

    # Obstacle events should have been published to the bus
    assert len(received) > 0, "Expected at least one CellUpdate from the Roomba bump"
    assert any(u.r_delta > 0 for u in received)
