"""Tests for EXP4 — Calibrated Policy Thresholds."""
from __future__ import annotations

import math

from manifold.experiments.calibrated_policy import (
    ThresholdCalibrator,
    run_calibration_benchmark,
)


def _calibrator_with_data(n: int = 150) -> ThresholdCalibrator:
    """Return a ThresholdCalibrator pre-loaded with synthetic outcome data."""
    import random
    rng = random.Random(2024)
    cal = ThresholdCalibrator()
    outcomes = []
    for _ in range(n):
        risk = rng.uniform(0.3, 1.0)
        correct = risk > 0.65 + rng.gauss(0, 0.05)
        outcomes.append({
            "risk_score": risk,
            "stakes": rng.uniform(0.4, 1.0),
            "action": "escalate" if risk > 0.7 else "proceed",
            "was_escalated": risk > 0.7,
            "was_correct_to_escalate": bool(correct),
            "domain": rng.choice(["finance", "medical", "legal"]),
        })
    cal._outcomes = outcomes
    return cal


# Test 1: platt_scale on perfectly correlated data returns meaningful a, b
def test_platt_scale_meaningful_params():
    cal = ThresholdCalibrator()
    # Perfectly separated: low scores → 0, high scores → 1
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1, 1]
    a, b = cal.platt_scale(scores, labels)
    assert math.isfinite(a)
    assert math.isfinite(b)
    # With higher scores predicting positive class, 'a' should be positive
    assert a > 0.0, f"Expected positive a (score predicts class), got {a}"


# Test 2: calibrate_threshold returns value in [0.0, 1.0]
def test_calibrate_threshold_in_range():
    cal = _calibrator_with_data()
    threshold = cal.calibrate_threshold(domain="all", target_precision=0.90)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0, f"Threshold {threshold} out of [0, 1]"


# Test 3: calibration_report returns dict with required keys
def test_calibration_report_required_keys():
    cal = _calibrator_with_data()
    report = cal.calibration_report()
    assert isinstance(report, dict)
    assert "status" in report
    if report["status"] == "ok":
        assert "domains" in report
        assert "total_events" in report
    else:
        assert "minimum_events_needed" in report


# Test 4: expected_calibration_error returns float >= 0
def test_expected_calibration_error_non_negative():
    cal = _calibrator_with_data()
    ece = cal.expected_calibration_error()
    assert isinstance(ece, float)
    assert math.isfinite(ece)
    assert ece >= 0.0, f"ECE {ece} should be >= 0"


# Test 5: With 0 vault events, calibration_report returns insufficient_data
def test_calibration_report_empty_vault():
    cal = ThresholdCalibrator()
    # No outcomes loaded
    report = cal.calibration_report()
    assert report.get("status") == "insufficient_data", (
        f"Expected 'insufficient_data', got: {report}"
    )
    assert "minimum_events_needed" in report
    assert report["minimum_events_needed"] == 100
