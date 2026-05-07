"""TimeSeriesEncoder — encodes numeric time-series data into EncodedTask."""
from __future__ import annotations

from statistics import mean

from manifold.encoder_v2 import EncodedTask


class TimeSeriesEncoder:
    """Encodes a list of float values into an EncodedTask."""

    def __init__(self, danger_threshold: float = 0.85) -> None:
        self.danger_threshold = danger_threshold

    def encode(self, values: list[float]) -> EncodedTask:
        if not values:
            return EncodedTask(cost=0.0, risk=0.0, neutrality=1.0, asset=0.0)
        risk = min(1.0, max(values) / self.danger_threshold)
        # early mean = first 20%, recent mean = last 20%
        n = len(values)
        early_n = max(1, n // 5)
        early_mean = mean(values[:early_n])
        recent_mean = mean(values[-early_n:])
        cost = min(1.0, abs(recent_mean - early_mean) * 2.0)
        asset = max(0.0, 1.0 - risk)
        neutrality = max(0.0, 1.0 - risk - cost)
        return EncodedTask(cost=cost, risk=risk, neutrality=neutrality, asset=asset)
