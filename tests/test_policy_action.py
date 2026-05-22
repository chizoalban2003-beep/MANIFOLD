"""Tests for the PolicyAction command interface on ManifoldBrain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from manifold.brain import BrainConfig, ManifoldBrain
from manifold.cell_update_bus import CellUpdate
from manifold.movement import MovementState
from manifold.policy_action import PolicyAction, PolicyCommandPayload


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


def test_handle_command_patrol_sets_movement_goal() -> None:
    brain = _make_brain()
    brain.handle_command(PolicyAction.PATROL, {"end": [5, 5, 0]})
    assert brain.movement.target_cell == (5, 5, 0)
    assert brain.movement.replan_pending is True


def test_handle_command_deploy_sets_goal() -> None:
    brain = _make_brain()
    brain.handle_command(PolicyAction.DEPLOY_AGENT, {"target": [3, 2, 1]})
    assert brain.movement.target_cell == (3, 2, 1)


def test_handle_command_emergency_stop_enters_error() -> None:
    brain = _make_brain()
    brain.handle_command(PolicyAction.EMERGENCY_STOP, {})
    assert brain.movement_state is MovementState.ERROR


def test_handle_command_unknown_code_no_crash() -> None:
    brain = _make_brain()
    brain.handle_command(999, {"foo": "bar"})
    # No exception raised, state unchanged
    assert brain.movement_state is MovementState.IDLE


def test_handle_command_return_home_default() -> None:
    brain = _make_brain()
    brain.handle_command(PolicyAction.RETURN_HOME, {})
    assert brain.movement.target_cell == (0, 0, 0)


def test_handle_command_defend_zone() -> None:
    brain = _make_brain()
    brain.handle_command(PolicyAction.DEFEND_ZONE, {"zone_center": [7, 7, 0]})
    assert brain.movement.target_cell == (7, 7, 0)


def test_policy_command_payload_model() -> None:
    p = PolicyCommandPayload(action_code=4, params={"end": [5, 5, 0]})
    assert p.action_code == 4
    assert p.params["end"] == [5, 5, 0]
    assert p.request_id is None


def test_policy_action_enum_values() -> None:
    assert PolicyAction.DEPLOY_AGENT == 1
    assert PolicyAction.EMERGENCY_STOP == 13
    assert len(PolicyAction) == 13


def test_gateway_routes_action_code_to_brain() -> None:
    """ManifoldMQTTGateway._on_command routes action_code payloads."""
    from manifold_physical.bridges.mqtt_bridge import ManifoldMQTTGateway, MQTTBridge

    brain = _make_brain()
    bridge = MQTTBridge.__new__(MQTTBridge)
    bridge.agent_id = "test-agent"
    bridge.command_callback = None
    bridge._mappings = {}
    bridge._running = False
    bridge._sock = None

    gateway = ManifoldMQTTGateway(brain=brain, bridge=bridge)
    # Simulate a command arriving
    gateway._on_command({"action_code": 4, "params": {"end": [4, 4, 0]}})
    assert brain.movement.target_cell == (4, 4, 0)
