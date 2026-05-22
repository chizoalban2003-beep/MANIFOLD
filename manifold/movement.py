"""Movement state machine for MANIFOLD agents.

The module keeps the movement logic separate from the decision logic so the
brain can compose it without coupling navigation to policy selection.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

from .cell_update_bus import CellUpdate

GridCoord = tuple[int, int, int]
FloatCoord = tuple[float, float, float]


class MovementState(IntEnum):
    """Discrete movement phases for a governed agent."""

    IDLE = 0
    TRANSITING = 1
    REPLANNING = 2
    ERROR = 3


@runtime_checkable
class PathPlanner(Protocol):
    """Minimal planner protocol used by the movement state machine."""

    def plan(self, start: GridCoord, target: GridCoord, **kwargs: Any) -> dict[str, Any]:
        """Return a planning result dictionary with a ``path`` key."""


def _manhattan(a: GridCoord, b: GridCoord) -> int:
    """Return the Manhattan distance between two 3-D grid coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _snap(coord: FloatCoord) -> GridCoord:
    """Snap a continuous coordinate to the nearest integer grid cell."""
    return (
        int(round(coord[0])),
        int(round(coord[1])),
        int(round(coord[2])),
    )


def _is_snapped(coord: FloatCoord, *, epsilon: float = 1e-9) -> bool:
    """Return ``True`` if *coord* is already perfectly aligned to the grid."""
    return all(abs(value - round(value)) <= epsilon for value in coord)


@dataclass(slots=True)
class MovementStateMachine:
    """Track navigation state, replanning requests, and path execution."""

    logical_pos: GridCoord = (0, 0, 0)
    physical_pos: FloatCoord = (0.0, 0.0, 0.0)
    target_cell: GridCoord | None = None
    next_safe_cell: GridCoord | None = None
    current_path: list[GridCoord] = field(default_factory=list)
    current_path_set: set[GridCoord] = field(default_factory=set)
    state: MovementState = MovementState.IDLE
    replan_pending: bool = False
    movement_speed: float = 1.0
    snap_epsilon: float = 1e-9
    replan_cooldown_seconds: float = 2.0
    min_shortcut_savings: int = 2

    _clear_replan_cooldown_until: float = field(default=0.0, repr=False)

    def set_goal(self, target_cell: GridCoord) -> None:
        """Store a new target cell and request a fresh plan."""
        self.target_cell = target_cell
        self.replan_pending = True

    def clear_goal(self) -> None:
        """Clear the active navigation target and return to idle."""
        self.target_cell = None
        self.current_path.clear()
        self.current_path_set.clear()
        self.next_safe_cell = None
        self.replan_pending = False
        self.state = MovementState.IDLE

    def on_cell_update(self, update: CellUpdate) -> None:
        """React to ``cell_blocked`` and ``cell_cleared`` bus events."""
        event = (update.reason or "").strip().lower()
        coord = (update.coord.x, update.coord.y, update.coord.z)
        now = float(update.coord.t or update.timestamp or time.monotonic())

        if event == "cell_blocked":
            self._handle_blocked(coord)
        elif event == "cell_cleared":
            self._handle_cleared(coord, now)

    def tick(self, delta_time: float, planner: PathPlanner | None = None) -> None:
        """Advance the movement state machine by *delta_time* seconds."""
        if delta_time <= 0.0:
            return

        if _is_snapped(self.physical_pos, epsilon=self.snap_epsilon):
            self.physical_pos = _snap(self.physical_pos)
            self.logical_pos = _snap(self.physical_pos)

        if self.state is MovementState.IDLE:
            if self.target_cell is not None and self.replan_pending:
                self._replan(planner)
            elif self.target_cell is not None and not self.current_path and self.next_safe_cell is None:
                self._replan(planner)
            return

        if self.state is MovementState.REPLANNING:
            if _is_snapped(self.physical_pos, epsilon=self.snap_epsilon):
                self._replan(planner)
            return

        self._transit(delta_time, planner)

    def _transit(self, delta_time: float, planner: PathPlanner | None) -> None:
        """Move physically toward ``next_safe_cell`` and snap on arrival."""
        remaining_time = delta_time
        while (
            remaining_time > 0.0
            and self.state is MovementState.TRANSITING
            and self.next_safe_cell is not None
        ):
            target = tuple(float(value) for value in self.next_safe_cell)
            distance = math.dist(self.physical_pos, target)
            if distance <= self.snap_epsilon:
                self._arrive_at_next_cell()
                if self.replan_pending:
                    self.state = MovementState.REPLANNING
                    if _is_snapped(self.physical_pos, epsilon=self.snap_epsilon):
                        self._replan(planner)
                    return
                continue

            max_step = self.movement_speed * remaining_time
            if max_step >= distance:
                remaining_time -= distance / max(self.movement_speed, 1e-9)
                self.physical_pos = target
                self._arrive_at_next_cell()
                if self.replan_pending:
                    self.state = MovementState.REPLANNING
                    if _is_snapped(self.physical_pos, epsilon=self.snap_epsilon):
                        self._replan(planner)
                    return
                continue

            ratio = max_step / distance
            self.physical_pos = tuple(
                current + (target[idx] - current) * ratio
                for idx, current in enumerate(self.physical_pos)
            )
            remaining_time = 0.0

    def _arrive_at_next_cell(self) -> None:
        """Commit the current step and advance to the next safe cell if any."""
        if self.next_safe_cell is None:
            self.state = MovementState.IDLE
            self.logical_pos = _snap(self.physical_pos)
            return

        self.logical_pos = self.next_safe_cell
        self.physical_pos = tuple(float(value) for value in self.next_safe_cell)

        if self.current_path and self.current_path[0] == self.logical_pos:
            self.current_path.pop(0)
        elif self.current_path:
            self.current_path = [self.logical_pos] + self.current_path[1:]

        self.current_path_set = set(self.current_path)
        self.next_safe_cell = self.current_path[1] if len(self.current_path) > 1 else None
        self.state = MovementState.TRANSITING if self.next_safe_cell is not None else MovementState.IDLE

    def _replan(self, planner: PathPlanner | None) -> bool:
        """Recompute the current path from the snapped logical position."""
        if self.target_cell is None:
            self.clear_goal()
            return False

        if planner is None:
            self.state = MovementState.IDLE
            return False

        result = planner.plan(start=self.logical_pos, target=self.target_cell)
        path = [tuple(int(value) for value in coord) for coord in result.get("path", [])]
        if not result.get("found") or not path:
            self.current_path = [self.logical_pos]
            self.current_path_set = {self.logical_pos}
            self.next_safe_cell = None
            self.replan_pending = False
            self.state = MovementState.IDLE
            return False

        if path[0] != self.logical_pos:
            path.insert(0, self.logical_pos)

        self.current_path = path
        self.current_path_set = set(path)
        self.next_safe_cell = path[1] if len(path) > 1 else None
        self.replan_pending = False
        self.state = MovementState.TRANSITING if self.next_safe_cell is not None else MovementState.IDLE
        return True

    def _handle_blocked(self, coord: GridCoord) -> None:
        """Mark a replan if an occupied cell lies on the active path."""
        if coord == self.next_safe_cell:
            return
        if coord in self.current_path_set:
            self.replan_pending = True
            if self.state is MovementState.IDLE:
                self.state = MovementState.REPLANNING

    def _handle_cleared(self, coord: GridCoord, current_time: float) -> None:
        """Trigger opportunistic replanning when a cleared shortcut is valuable."""
        if current_time < self._clear_replan_cooldown_until:
            return
        if self.target_cell is None:
            return

        if coord in self.current_path_set:
            current_remaining = max(0, len(self.current_path) - 1)
            via_cleared = _manhattan(self.logical_pos, coord) + _manhattan(coord, self.target_cell)
            if current_remaining - via_cleared >= self.min_shortcut_savings:
                self.replan_pending = True
                self._clear_replan_cooldown_until = current_time + self.replan_cooldown_seconds
                if self.state is MovementState.IDLE:
                    self.state = MovementState.REPLANNING


# ---------------------------------------------------------------------------
# Watchdog — hardware heartbeat safety monitor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Watchdog:
    """Heartbeat watchdog that triggers EMERGENCY_STOP on timeout.

    The brain must call :meth:`feed` at the configured interval (default 500 ms).
    If :meth:`is_expired` returns ``True``, the control loop must halt the robot
    and transition to ``MovementState.ERROR``.  Once :meth:`feed` is called again
    the watchdog recovers automatically (``is_expired`` returns ``False``).

    Parameters
    ----------
    timeout_seconds:
        Maximum allowed time between heartbeat feeds (default: 0.5 s).
    """

    timeout_seconds: float = 0.5
    _last_fed: float = field(default_factory=time.monotonic, repr=False)

    def feed(self, current_time: float | None = None) -> None:
        """Reset the watchdog timer."""
        self._last_fed = float(current_time if current_time is not None else time.monotonic())

    def is_expired(self, current_time: float | None = None) -> bool:
        """Return ``True`` if the heartbeat has timed out."""
        now = float(current_time if current_time is not None else time.monotonic())
        return (now - self._last_fed) >= self.timeout_seconds

    @property
    def elapsed(self) -> float:
        """Seconds since the last feed."""
        return time.monotonic() - self._last_fed
