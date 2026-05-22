"""Tests for the hysteresis-based DynamicGridManager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from manifold.cell_update_bus import CellUpdate
from manifold.dynamic_grid import DynamicGridManager, OccupancyState


@dataclass
class _BusStub:
    subscribers: dict[str, Callable[[CellUpdate], None]] = field(default_factory=dict)
    events: list[CellUpdate] = field(default_factory=list)

    def subscribe(self, subscriber_id: str, callback: Callable[[CellUpdate], None]) -> None:
        self.subscribers[subscriber_id] = callback

    def publish(self, update: CellUpdate) -> None:
        self.events.append(update)


def test_dynamic_grid_manager_emits_one_event_per_threshold_crossing() -> None:
    bus = _BusStub()
    manager = DynamicGridManager(
        bus=bus,
        hit_confidence_delta=0.4,
        decay_rate_per_second=0.1,
    )

    cell = manager.register_sensor_hit((2, 3, 0), current_time=0.0)
    assert cell.state is OccupancyState.PENDING
    assert bus.events == []

    cell = manager.register_sensor_hit((2, 3, 0), current_time=0.0)
    assert cell.state is OccupancyState.DYNAMIC_BLOCK
    assert len(bus.events) == 1
    assert bus.events[0].reason == "cell_blocked"

    manager.register_sensor_hit((2, 3, 0), current_time=0.0)
    assert len(bus.events) == 1

    manager.tick_decay(3.0)
    assert len(bus.events) == 1
    assert manager.get_cell((2, 3, 0)).state is OccupancyState.DYNAMIC_BLOCK

    manager.tick_decay(7.2)
    assert len(bus.events) == 2
    assert bus.events[1].reason == "cell_cleared"
    assert manager.get_cell((2, 3, 0)).state is OccupancyState.PENDING
