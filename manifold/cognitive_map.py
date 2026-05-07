"""CognitiveMap — spatial navigation and outcome memory for ManifoldBrain."""
from __future__ import annotations

import math
from typing import Any


class CognitiveMap:
    """Navigate GridWorld cells and learn from past outcomes."""

    def __init__(self) -> None:
        # key: (row, col) -> list of outcome dicts
        self._outcome_log: dict[tuple[int, int], list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    def navigate(self, query: Any, world: Any, k: int = 3) -> list[dict]:
        """Return k nearest grid cells sorted ascending by Euclidean distance."""
        qv = (query.cost, query.risk, query.neutrality, query.asset)
        results = []
        for row, row_cells in enumerate(world.cells):
            for col, cell in enumerate(row_cells):
                cv = (cell.cost, cell.risk, cell.neutrality, cell.asset)
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(qv, cv)))
                # Find most recent past_action for this cell
                past_action = None
                history = self._outcome_log.get((row, col), [])
                if history:
                    past_action = history[-1]["action"]
                results.append(
                    {
                        "row": row,
                        "col": col,
                        "distance": dist,
                        "cell": cell,
                        "past_action": past_action,
                    }
                )
        results.sort(key=lambda r: r["distance"])
        return results[:k]

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        row: int,
        col: int,
        action: str,
        success: bool,
        risk_score: float,
    ) -> None:
        """Log a task outcome at a grid position."""
        key = (row, col)
        if key not in self._outcome_log:
            self._outcome_log[key] = []
        self._outcome_log[key].append(
            {"action": action, "success": success, "risk_score": risk_score}
        )

    # ------------------------------------------------------------------
    def suggest_action(self, query: Any, world: Any, fallback: str = "verify") -> str:
        """Return the past_action from nearest successful outcome, or fallback."""
        neighbours = self.navigate(query, world, k=len(world.cells) * len(world.cells[0]))
        for neighbour in neighbours:
            key = (neighbour["row"], neighbour["col"])
            for entry in reversed(self._outcome_log.get(key, [])):
                if entry["success"]:
                    return entry["action"]
        return fallback
