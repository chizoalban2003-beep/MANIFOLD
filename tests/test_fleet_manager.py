"""Tests for the Fleet Orchestrator (Town Hall) pattern on ManifoldBrain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from manifold.brain import BrainConfig, ManifoldBrain
from manifold.cell_update_bus import CellUpdate
from manifold.movement import MovementState
from manifold.policy_action import PolicyAction


@dataclass
class _BusStub:
    subscribers: dict[str, Callable[[CellUpdate], None]] = field(default_factory=dict)

    def subscribe(self, subscriber_id: str, callback: Callable[[CellUpdate], None]) -> None:
        self.subscribers[subscriber_id] = callback

    def publish(self, update: CellUpdate) -> None:
        for callback in self.subscribers.values():
            callback(update)


class _PlannerStub:
    def __init__(self, path: list[tuple[int, int, int]] | None = None) -> None:
        self.path = path or [(1, 0, 0), (2, 0, 0), (3, 0, 0)]

    def plan(self, start: tuple[int, int, int], target: tuple[int, int, int], **kwargs):
        path = [start, *[c for c in self.path if c not in {start, target}], target]
        return {"found": True, "path": path, "total_cost": float(len(path) - 1), "total_risk": 0.0}


def _make_brain() -> ManifoldBrain:
    bus = _BusStub()
    planner = _PlannerStub()
    brain = ManifoldBrain(BrainConfig(), tools=[], movement_planner=planner, movement_bus=bus)
    brain.logical_pos = (0, 0, 0)
    brain.physical_pos = (0.0, 0.0, 0.0)
    return brain


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


def test_register_agent_creates_fleet_entries() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.register_agent("robot-2")

    assert "robot-1" in brain.fleet
    assert "robot-2" in brain.fleet
    assert "robot-1" in brain.fleet_watchdogs
    assert "robot-2" in brain.fleet_watchdogs
    assert len(brain.fleet) == 2


def test_register_agent_is_idempotent() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    m1 = brain.fleet["robot-1"]
    brain.register_agent("robot-1")  # second call — no-op
    assert brain.fleet["robot-1"] is m1
    assert len(brain.fleet) == 1


def test_unregister_agent_removes_entries() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.unregister_agent("robot-1")
    assert "robot-1" not in brain.fleet
    assert "robot-1" not in brain.fleet_watchdogs


def test_fleet_agent_ids() -> None:
    brain = _make_brain()
    brain.register_agent("a")
    brain.register_agent("b")
    ids = brain.fleet_agent_ids()
    assert set(ids) == {"a", "b"}


# ------------------------------------------------------------------
# Targeted command routing
# ------------------------------------------------------------------


def test_handle_command_targeted_to_fleet_agent() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.handle_command(PolicyAction.PATROL, {"end": [5, 5, 0]}, agent_id="robot-1")
    # Fleet agent gets the goal
    assert brain.fleet["robot-1"].target_cell == (5, 5, 0)
    # Primary agent is NOT affected
    assert brain.movement.target_cell is None


def test_handle_command_broadcast_all() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.register_agent("robot-2")
    brain.handle_command(PolicyAction.RETURN_HOME, {"home": [0, 0, 0]}, agent_id="ALL")
    # Primary and all fleet agents get the goal
    assert brain.movement.target_cell == (0, 0, 0)
    assert brain.fleet["robot-1"].target_cell == (0, 0, 0)
    assert brain.fleet["robot-2"].target_cell == (0, 0, 0)


def test_handle_command_no_agent_id_goes_to_primary() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.handle_command(PolicyAction.DEPLOY_AGENT, {"target": [3, 2, 1]})
    # Primary gets it
    assert brain.movement.target_cell == (3, 2, 1)
    # Fleet agent does NOT get it (no broadcast without "ALL")
    assert brain.fleet["robot-1"].target_cell is None


# ------------------------------------------------------------------
# Fleet tick — per-agent safety isolation
# ------------------------------------------------------------------


def test_fleet_tick_advances_all_agents() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.fleet["robot-1"].set_goal((3, 0, 0))

    # Feed all watchdogs so nothing expires
    brain.feed_watchdog()
    brain.feed_watchdog(agent_id="robot-1")

    # Tick multiple times to allow planning + transit (speed=1.0, 3 cells)
    for _ in range(10):
        brain.feed_watchdog()
        brain.feed_watchdog(agent_id="robot-1")
        brain.tick(1.0)

    # Fleet agent should have reached target or at least moved
    assert brain.fleet["robot-1"].logical_pos != (0, 0, 0)


def test_fleet_watchdog_isolation() -> None:
    """If one fleet agent's watchdog expires, only THAT agent enters ERROR."""
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.register_agent("robot-2")

    # Feed primary and robot-2 — keep them alive
    brain.feed_watchdog()
    brain.feed_watchdog(agent_id="robot-2")

    # Expire robot-1's watchdog
    brain.fleet_watchdogs["robot-1"]._last_fed = 0.0

    brain.tick(0.1)

    # robot-1 enters ERROR; robot-2 and primary are fine
    assert brain.fleet["robot-1"].state is MovementState.ERROR
    assert brain.fleet["robot-2"].state is not MovementState.ERROR
    assert brain.movement.state is not MovementState.ERROR


def test_fleet_watchdog_recovery() -> None:
    """Feeding a fleet agent's watchdog recovers it from ERROR on next tick."""
    brain = _make_brain()
    brain.register_agent("robot-1")

    # Expire and tick
    brain.fleet_watchdogs["robot-1"]._last_fed = 0.0
    brain.feed_watchdog()  # keep primary alive
    brain.tick(0.1)
    assert brain.fleet["robot-1"].state is MovementState.ERROR

    # Feed robot-1 and tick again
    brain.feed_watchdog(agent_id="robot-1")
    brain.tick(0.1)
    assert brain.fleet["robot-1"].state is MovementState.IDLE


# ------------------------------------------------------------------
# Emergency stop — targeted
# ------------------------------------------------------------------


def test_emergency_stop_targeted_fleet_agent() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.handle_command(PolicyAction.EMERGENCY_STOP, {}, agent_id="robot-1")
    assert brain.fleet["robot-1"].state is MovementState.ERROR
    # Primary is not affected
    assert brain.movement.state is not MovementState.ERROR


def test_emergency_stop_primary_when_no_agent_id() -> None:
    brain = _make_brain()
    brain.register_agent("robot-1")
    brain.handle_command(PolicyAction.EMERGENCY_STOP, {})
    assert brain.movement.state is MovementState.ERROR
    # Fleet agent is not affected
    assert brain.fleet["robot-1"].state is not MovementState.ERROR
