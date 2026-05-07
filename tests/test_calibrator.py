"""Tests for manifold/calibrator.py"""
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock
from manifold.calibrator import calibrate_domain, run_calibration, calibration_report, MIN_SAMPLES


def test_insufficient_samples_returns_not_adjusted():
    db = AsyncMock()
    db.get_domain_stats.return_value = {"total_tasks": 10, "total_escalations": 2}
    result = asyncio.run(calibrate_domain(db, "general"))
    assert result.adjusted is False
    assert result.samples == 10


def test_over_escalating_raises_threshold():
    db = AsyncMock()
    # actual_rate = 40/100 = 0.40, target = 0.08 → error = 0.32 → adjustment positive
    db.get_domain_stats.return_value = {"total_tasks": 100, "total_escalations": 40}
    result = asyncio.run(calibrate_domain(db, "general"))
    assert result.new_threshold > result.old_threshold


def test_under_escalating_lowers_threshold():
    db = AsyncMock()
    # actual_rate = 0/100 = 0.0, target = 0.08 → error = -0.08 → adjustment negative
    db.get_domain_stats.return_value = {"total_tasks": 100, "total_escalations": 0}
    result = asyncio.run(calibrate_domain(db, "general"))
    assert result.new_threshold < result.old_threshold


def test_threshold_clamps_to_bounds():
    db = AsyncMock()
    # Extreme over-escalation
    db.get_domain_stats.return_value = {"total_tasks": 1000, "total_escalations": 1000}
    result = asyncio.run(calibrate_domain(db, "general"))
    assert result.new_threshold <= 0.95
    # Extreme under-escalation
    db.get_domain_stats.return_value = {"total_tasks": 1000, "total_escalations": 0}
    result2 = asyncio.run(calibrate_domain(db, "general"))
    assert result2.new_threshold >= 0.05


def test_calibration_report_returns_string():
    db = AsyncMock()
    db.get_domain_stats.return_value = {"total_tasks": 10, "total_escalations": 0}
    report = asyncio.run(calibration_report(db))
    assert isinstance(report, str)
    assert len(report) > 0
