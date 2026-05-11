"""Tests for manifold.cell_update_bus."""
import threading
import time

import pytest

from manifold.cell_update_bus import CellCoord, CellUpdate, CellUpdateBus


@pytest.fixture()
def bus():
    """Fresh CellUpdateBus for each test."""
    return CellUpdateBus()


def test_subscribe_and_publish_calls_callback(bus):
    received = []
    bus.subscribe("test-sub", lambda u: received.append(u))
    update = CellUpdate(coord=CellCoord(x=1, y=2, z=0), r_delta=0.5, reason="test")
    bus.publish(update)
    # Give background thread time to deliver
    time.sleep(0.05)
    assert len(received) == 1
    assert received[0].r_delta == 0.5


def test_multiple_subscribers_all_receive(bus):
    results = {"a": [], "b": [], "c": []}
    for k in results:
        bus.subscribe(k, lambda u, key=k: results[key].append(u))
    bus.publish(CellUpdate(coord=CellCoord(x=0, y=0), r_delta=0.3))
    time.sleep(0.05)
    for k, v in results.items():
        assert len(v) == 1, f"subscriber {k} did not receive update"


def test_recent_returns_published_updates(bus):
    for i in range(5):
        bus.publish(CellUpdate(coord=CellCoord(x=i, y=0), r_delta=float(i) * 0.1))
    time.sleep(0.02)
    recent = bus.recent()
    assert len(recent) == 5
    coords = [u.coord.x for u in recent]
    assert coords == [0, 1, 2, 3, 4]


def test_publish_obstacle_fires_correctly(bus):
    received = []
    bus.subscribe("obs", lambda u: received.append(u))
    bus.publish_obstacle(3, 4, 0, risk_spike=0.9, source="lidar-01", ttl=30.0, reason="cat")
    time.sleep(0.05)
    assert len(received) == 1
    u = received[0]
    assert u.coord.x == 3
    assert u.coord.y == 4
    assert u.r_delta == 0.9
    assert u.source == "lidar-01"
    assert u.reason == "cat"


def test_unsubscribe_stops_delivery(bus):
    received = []
    bus.subscribe("unsub-test", lambda u: received.append(u))
    bus.publish(CellUpdate(coord=CellCoord(x=0, y=0), r_delta=0.1))
    time.sleep(0.05)
    assert len(received) == 1
    bus.unsubscribe("unsub-test")
    bus.publish(CellUpdate(coord=CellCoord(x=1, y=1), r_delta=0.2))
    time.sleep(0.05)
    # Still only 1 — the second publish was not delivered
    assert len(received) == 1
