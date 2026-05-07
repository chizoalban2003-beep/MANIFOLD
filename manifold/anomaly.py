"""Phase 68: Sliding-window statistical anomaly detector for tool behaviour."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# ToolBehaviourWindow
# ---------------------------------------------------------------------------

@dataclass
class ToolBehaviourWindow:
    tool_name: str
    _scores: deque = field(default_factory=lambda: deque(maxlen=100), init=False, repr=False)

    def record(self, success: bool) -> None:
        self._scores.append(1.0 if success else 0.0)

    @property
    def mean(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    @property
    def std(self) -> float:
        if len(self._scores) < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._scores) / len(self._scores)
        return math.sqrt(variance)

    @property
    def recent_mean(self) -> float:
        recent = list(self._scores)[-20:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def z_score_anomaly(self) -> float:
        if len(self._scores) < 30:
            return 0.0
        s = self.std
        if s < 0.001:
            return 0.0
        return (self.mean - self.recent_mean) / s

    def is_anomalous(self, z_threshold: float = 2.0) -> bool:
        return self.z_score_anomaly() >= z_threshold

# ---------------------------------------------------------------------------
# ManifoldAnomalyDetector
# ---------------------------------------------------------------------------

class ManifoldAnomalyDetector:
    def __init__(self, z_threshold: float = 2.0, window_size: int = 100) -> None:
        self._z_threshold = z_threshold
        self._window_size = window_size
        self._tools: dict[str, ToolBehaviourWindow] = {}

    def _get_window(self, tool_name: str) -> ToolBehaviourWindow:
        if tool_name not in self._tools:
            w = ToolBehaviourWindow(tool_name=tool_name)
            w._scores = deque(maxlen=self._window_size)
            self._tools[tool_name] = w
        return self._tools[tool_name]

    def record_outcome(self, tool_name: str, success: bool, timestamp: Any = None) -> None:
        self._get_window(tool_name).record(success)

    def is_anomalous(self, tool_name: str) -> bool:
        if tool_name not in self._tools:
            return False
        return self._tools[tool_name].is_anomalous(self._z_threshold)

    def anomaly_score(self, tool_name: str) -> float:
        if tool_name not in self._tools:
            return 0.0
        return self._tools[tool_name].z_score_anomaly()

    def all_anomalous_tools(self) -> list[str]:
        return [name for name in self._tools if self.is_anomalous(name)]

    def summary(self) -> dict[str, dict]:
        result = {}
        for name, window in self._tools.items():
            result[name] = {
                "mean": window.mean,
                "recent_mean": window.recent_mean,
                "z_score": window.z_score_anomaly(),
                "anomalous": window.is_anomalous(self._z_threshold),
                "samples": len(window._scores),
            }
        return result
