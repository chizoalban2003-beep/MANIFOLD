"""Tests for manifold/anomaly.py"""
from __future__ import annotations
import pytest
from manifold.anomaly import ManifoldAnomalyDetector, ToolBehaviourWindow


def test_stable_tool_not_anomalous():
    det = ManifoldAnomalyDetector(z_threshold=2.0)
    for _ in range(60):
        det.record_outcome("stable_api", success=True)
    assert det.is_anomalous("stable_api") is False


def test_degraded_tool_is_anomalous():
    det = ManifoldAnomalyDetector(z_threshold=1.5)
    for _ in range(60):
        det.record_outcome("api", success=True)
    for _ in range(20):
        det.record_outcome("api", success=False)
    assert det.is_anomalous("api") is True


def test_unknown_tool_not_anomalous():
    det = ManifoldAnomalyDetector()
    assert det.is_anomalous("nonexistent") is False
    assert det.anomaly_score("nonexistent") == 0.0


def test_z_score_increases_after_degradation():
    det = ManifoldAnomalyDetector()
    for _ in range(60):
        det.record_outcome("tool", success=True)
    score_before = det.anomaly_score("tool")
    for _ in range(20):
        det.record_outcome("tool", success=False)
    score_after = det.anomaly_score("tool")
    assert score_after > score_before


def test_summary_contains_all_tools():
    det = ManifoldAnomalyDetector()
    det.record_outcome("tool_a", success=True)
    det.record_outcome("tool_b", success=False)
    summary = det.summary()
    assert "tool_a" in summary
    assert "tool_b" in summary
