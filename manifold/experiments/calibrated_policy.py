"""manifold/experiments/calibrated_policy.py — Per-domain threshold calibration."""

from __future__ import annotations

import math
import random
from typing import Sequence


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class ThresholdCalibrator:
    """Fits decision thresholds from outcome history using Platt scaling and ECE."""

    _MIN_EVENTS = 100
    _DEFAULT_THRESHOLD = 0.7

    def __init__(self) -> None:
        self._outcomes: list[dict] = []

    def platt_scale(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        lr: float = 0.5,
        n_iter: int = 1000,
    ) -> tuple[float, float]:
        """Fit logistic P(y=1|score) = sigmoid(a*score + b). Returns (a, b)."""
        a, b = 0.0, 0.0
        n = len(scores)
        if n == 0:
            return a, b
        for _ in range(n_iter):
            da, db = 0.0, 0.0
            for s, y in zip(scores, labels):
                pred = _sigmoid(a * s + b)
                err = pred - y
                da += err * s
                db += err
            a -= lr * da / n
            b -= lr * db / n
        return a, b

    def calibrate_threshold(
        self,
        domain: str = "all",
        target_precision: float = 0.90,
    ) -> float:
        """Find threshold achieving target_precision for escalation decisions."""
        outcomes = (
            self._outcomes
            if domain == "all"
            else [o for o in self._outcomes if o.get("domain") == domain]
        )
        if not outcomes:
            return self._DEFAULT_THRESHOLD

        sorted_o = sorted(
            outcomes, key=lambda o: o.get("risk_score", 0.0), reverse=True
        )
        for i in range(1, len(sorted_o) + 1):
            subset = sorted_o[:i]
            tp = sum(1 for o in subset if o.get("was_correct_to_escalate", False))
            precision = tp / len(subset)
            if precision >= target_precision:
                threshold = sorted_o[i - 1].get("risk_score", self._DEFAULT_THRESHOLD)
                return float(max(0.0, min(1.0, threshold)))
        return self._DEFAULT_THRESHOLD

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """Compute ECE: weighted avg |mean_score - fraction_positive| per bin."""
        if not self._outcomes:
            return 0.0
        bins: list[list[dict]] = [[] for _ in range(n_bins)]
        for o in self._outcomes:
            score = o.get("risk_score", 0.0)
            idx = min(int(score * n_bins), n_bins - 1)
            bins[idx].append(o)
        ece = 0.0
        n = len(self._outcomes)
        for b in bins:
            if not b:
                continue
            mean_score = sum(o.get("risk_score", 0.0) for o in b) / len(b)
            frac_pos = (
                sum(1 for o in b if o.get("was_correct_to_escalate", False)) / len(b)
            )
            ece += (len(b) / n) * abs(mean_score - frac_pos)
        return float(ece)

    def calibration_report(self) -> dict:
        """Return calibration status report."""
        n = len(self._outcomes)
        if n < self._MIN_EVENTS:
            return {
                "status": "insufficient_data",
                "minimum_events_needed": self._MIN_EVENTS,
            }
        domains = list({o.get("domain", "unknown") for o in self._outcomes})
        return {
            "status": "ok",
            "total_events": n,
            "domains": domains,
        }


def run_calibration_benchmark(seed: int = 42, n_events: int = 300) -> dict:
    """Run synthetic calibration benchmark. Returns metrics dict."""
    rng = random.Random(seed)
    cal = ThresholdCalibrator()
    domains = ["finance", "medical", "legal"]
    outcomes = []
    for _ in range(n_events):
        risk = rng.uniform(0.3, 1.0)
        correct = risk > 0.65 + rng.gauss(0, 0.05)
        outcomes.append({
            "risk_score": risk,
            "stakes": rng.uniform(0.4, 1.0),
            "action": "escalate" if risk > 0.7 else "proceed",
            "was_escalated": risk > 0.7,
            "was_correct_to_escalate": bool(correct),
            "domain": rng.choice(domains),
        })
    cal._outcomes = outcomes
    threshold = cal.calibrate_threshold(domain="all", target_precision=0.85)
    ece = cal.expected_calibration_error()
    report = cal.calibration_report()
    return {
        "threshold": threshold,
        "ece": ece,
        "report": report,
        "n_events": n_events,
    }
