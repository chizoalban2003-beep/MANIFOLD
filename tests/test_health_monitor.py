"""Tests for manifold.health_monitor."""
import time

import pytest

from manifold.cell_update_bus import CellUpdateBus
from manifold.health_monitor import DigitalHealthMonitor, ToolHealth


@pytest.fixture()
def monitor_and_bus():
    """Fresh monitor + bus for each test."""
    bus = CellUpdateBus()
    monitor = DigitalHealthMonitor()
    # Patch monitor to use our test bus
    import manifold.health_monitor as hm_mod
    old_get = hm_mod.get_bus
    hm_mod.get_bus = lambda: bus
    yield monitor, bus
    hm_mod.get_bus = old_get


def test_record_outcome_success_increments_success_count(monitor_and_bus):
    monitor, _bus = monitor_and_bus
    monitor.register_tool("tool-a")
    monitor.record_outcome("tool-a", success=True, latency_ms=50.0)
    status = monitor.status()
    assert status["tool-a"]["success_count"] == 1
    assert status["tool-a"]["error_count"] == 0


def test_high_error_rate_triggers_cellupdate(monitor_and_bus):
    monitor, bus = monitor_and_bus
    received = []
    bus.subscribe("test-health", lambda u: received.append(u))
    monitor.register_tool("tool-b", grid_coord=(1, 1, 3))
    # 6 errors, 1 success → error_rate ≈ 0.86 > 0.8 → degraded
    for _ in range(6):
        monitor.record_outcome("tool-b", success=False)
    monitor.record_outcome("tool-b", success=True)
    time.sleep(0.1)
    assert len(received) > 0
    r_deltas = [u.r_delta for u in received]
    assert any(d > 0 for d in r_deltas)


def test_record_rate_limit_fires_cellupdate_with_c_delta(monitor_and_bus):
    monitor, bus = monitor_and_bus
    received = []
    bus.subscribe("test-ratelimit", lambda u: received.append(u))
    monitor.register_tool("tool-c", grid_coord=(2, 2, 3))
    monitor.record_rate_limit("tool-c", retry_after_seconds=60.0)
    time.sleep(0.1)
    assert len(received) > 0
    assert any(u.c_delta > 0 for u in received)


def test_record_recovery_publishes_negative_r_delta(monitor_and_bus):
    monitor, bus = monitor_and_bus
    received = []
    bus.subscribe("test-recovery", lambda u: received.append(u))
    monitor.register_tool("tool-d", grid_coord=(3, 3, 3))
    monitor.record_recovery("tool-d")
    time.sleep(0.1)
    assert len(received) > 0
    assert any(u.r_delta < 0 for u in received)


def test_status_returns_dict_with_tool_entries(monitor_and_bus):
    monitor, _bus = monitor_and_bus
    monitor.register_tool("tool-e")
    monitor.register_tool("tool-f")
    status = monitor.status()
    assert "tool-e" in status
    assert "tool-f" in status
    assert "error_rate" in status["tool-e"]
    assert "status" in status["tool-e"]
