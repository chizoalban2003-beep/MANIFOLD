"""StructuredEncoder — encodes a dict row into an EncodedTask."""
from __future__ import annotations

from statistics import mean

from manifold.encoder_v2 import EncodedTask

_DEFAULT_RISK_FIELDS = ["risk_score", "fraud_score", "severity", "error_rate"]
_DEFAULT_ASSET_FIELDS = ["value", "revenue", "customer_value", "priority"]


class StructuredEncoder:
    """Encodes a dict row into an EncodedTask based on labelled fields."""

    def __init__(
        self,
        risk_fields: list[str] | None = None,
        asset_fields: list[str] | None = None,
    ) -> None:
        self.risk_fields = risk_fields if risk_fields is not None else list(_DEFAULT_RISK_FIELDS)
        self.asset_fields = asset_fields if asset_fields is not None else list(_DEFAULT_ASSET_FIELDS)

    def encode(self, row: dict) -> EncodedTask:
        risk_vals = [float(row[f]) for f in self.risk_fields if f in row]
        asset_vals = [float(row[f]) for f in self.asset_fields if f in row]
        risk = mean(risk_vals) if risk_vals else 0.0
        asset = mean(asset_vals) if asset_vals else 0.5
        cost = min(1.0, risk * 0.5)
        neutrality = max(0.0, 1.0 - risk - cost)
        # clamp all
        risk = max(0.0, min(1.0, risk))
        asset = max(0.0, min(1.0, asset))
        cost = max(0.0, min(1.0, cost))
        neutrality = max(0.0, min(1.0, neutrality))
        return EncodedTask(cost=cost, risk=risk, neutrality=neutrality, asset=asset)
