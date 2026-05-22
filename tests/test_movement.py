"""Tests for MANIFOLD movement state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from manifold.brain import BrainConfig, ManifoldBrain
from manifold.cell_update_bus import CellCoord, CellUpdate
from manifold.movement import MovementState


@dataclass
class _BusStub:
    subscribers: dict[str, Callable[[CellUpdate], None]] = field(default_factory=dict)

    def subscribe(self, subscriber_id: str, callback: Callable[[CellUpdate], None]) -> None:
        self.subscribers[subscriber_id] = callback

    def publish(self, update: CellUpdate) -> None:
        for callback in self.subscribers.values():
            callback(update)


class _PlannerStub:
    def __init__(self, path: list[tuple[int, int, int]]) -> None:
        self.path = path

    def plan(self, start: tuple[int, int, int], target: tuple[int, int, int], **kwargs):
        path = [start, *[coord for coord in self.path if coord not in {start, target}], target]
        return {"found": True, "path": path, "total_cost": float(len(path) - 1), "total_risk": 0.0}


def test_brain_replans_after_safe_cell_arrival_when_blocked() -> None:
    bus = _BusStub()
    planner = _PlannerStub(path=[(1, 0, 0), (2, 0, 0), (3, 0, 0)])
    brain = ManifoldBrain(BrainConfig(), tools=[], movement_planner=planner, movement_bus=bus)

    brain.logical_pos = (0, 0, 0)
    brain.physical_pos = (0.0, 0.0, 0.0)
    brain.set_movement_goal((4, 0, 0))
    brain.tick(0.1)

    assert brain.movement_state is MovementState.TRANSITING
    assert brain.next_safe_cell == (1, 0, 0)

    bus.publish(CellUpdate(coord=CellCoord(x=2, y=0, z=0, t=0.0), reason="cell_blocked"))
    assert brain.replan_pending is True
    assert brain.movement_state is MovementState.TRANSITING

    brain.tick(1.0)
    assert brain.logical_pos == (1, 0, 0)
    assert brain.physical_pos == (1.0, 0.0, 0.0)
    assert brain.movement_state is MovementState.TRANSITING


def test_brain_cleared_shortcut_respects_distance_check_and_cooldown() -> None:
    bus = _BusStub()
    planner = _PlannerStub(path=[(0, 1, 0), (1, 1, 0), (1, 2, 0), (2, 2, 0), (3, 2, 0), (4, 2, 0), (4, 1, 0)])
    brain = ManifoldBrain(BrainConfig(), tools=[], movement_planner=planner, movement_bus=bus)

    brain.logical_pos = (0, 0, 0)
    brain.physical_pos = (0.0, 0.0, 0.0)
    brain.set_movement_goal((4, 0, 0))
    brain.movement.current_path = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 2, 0),
        (2, 2, 0),
        (3, 2, 0),
        (4, 2, 0),
        (4, 1, 0),
        (4, 0, 0),
    ]
    brain.movement.current_path_set = set(brain.movement.current_path)
    brain.movement.next_safe_cell = (0, 1, 0)
    brain.movement.state = MovementState.TRANSITING

    bus.publish(CellUpdate(coord=CellCoord(x=1, y=1, z=0, t=10.0), reason="cell_cleared"))
    assert brain.replan_pending is True

    brain.replan_pending = False
    bus.publish(CellUpdate(coord=CellCoord(x=1, y=1, z=0, t=10.5), reason="cell_cleared"))
    assert brain.replan_pending is False

    bus.publish(CellUpdate(coord=CellCoord(x=1, y=1, z=0, t=12.5), reason="cell_cleared"))
    assert brain.replan_pending is True
