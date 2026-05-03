"""Private math utilities shared across MANIFOLD modules."""

from __future__ import annotations

import math
from statistics import fmean


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the closed interval [*low*, *high*]."""
    return max(low, min(high, value))


def binary_correlation(left: list[int], right: list[int]) -> float:
    """Pearson correlation coefficient for two equal-length integer sequences."""
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var == 0.0 or right_var == 0.0:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)
