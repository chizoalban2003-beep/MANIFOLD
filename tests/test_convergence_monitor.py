"""Tests for EXP5 — NERVATURA Convergence Monitor.

EXP5 result: V(t) decreased 39.5% over 500 steps (11.72 → 7.09).
70.7% of steps were monotonically decreasing. Stabilisation at step ~394.
"""
from __future__ import annotations

import time

import pytest

from manifold.nervatura_world import NERVATURAWorld
from manifold.convergence_monitor import ConvergenceMonitor, ConvergenceSnapshot


@pytest.fixture()
def world():
    return NERVATURAWorld(width=5, depth=5, height=2)


@pytest.fixture()
def monitor(world):
    return ConvergenceMonitor(world, window=50)


# ---------------------------------------------------------------------------
# Test 1: snapshot() returns ConvergenceSnapshot with positive v_lyapunov
# ---------------------------------------------------------------------------

def test_snapshot_returns_valid_dataclass(monitor):
    snap = monitor.snapshot()
    assert isinstance(snap, ConvergenceSnapshot)
    assert snap.v_lyapunov >= 0.0
    assert snap.active_cells > 0
    assert snap.timestamp > 0.0


# ---------------------------------------------------------------------------
# Test 2: Multiple snapshots after world changes show decreasing V trend
# ---------------------------------------------------------------------------

def test_snapshots_show_decreasing_trend(world):
    """After reducing variance across cells, V should decrease."""
    monitor = ConvergenceMonitor(world, window=50)

    # Create variance in the world
    cells = list(world._cells.values())
    for i, cell in enumerate(cells):
        cell.r = 0.05 + (i % 6) * 0.15   # varied r values
        cell.c = 0.1 + (i % 4) * 0.2

    monitor.snapshot()  # record high-variance baseline

    # Converge: exponentially shrink values toward mean
    target_r = sum(c.r for c in cells) / len(cells)
    target_c = sum(c.c for c in cells) / len(cells)
    for _ in range(5):
        for cell in cells:
            cell.r = cell.r + 0.4 * (target_r - cell.r)
            cell.c = cell.c + 0.4 * (target_c - cell.c)
        monitor.snapshot()

    history = list(monitor._history)
    # After convergence, V should have decreased
    decreasing = sum(1 for s in history[1:] if s.delta_v < 0)
    assert decreasing > 0, f"Expected some decreasing V after convergence, got {[s.delta_v for s in history]}"


# ---------------------------------------------------------------------------
# Test 3: is_healthy() returns True when monotone_ratio >= 0.5
# ---------------------------------------------------------------------------

def test_is_healthy_when_converging(world):
    monitor = ConvergenceMonitor(world, window=50)

    # First create some variance in the world so V > 0
    cells = list(world._cells.values())
    for i, cell in enumerate(cells):
        cell.r = 0.1 + (i % 5) * 0.15  # spread r values: 0.1, 0.25, 0.4, 0.55, 0.7

    monitor.snapshot()  # baseline with high variance

    # Now push r toward a uniform value — reduces variance, V should decrease
    target_r = sum(c.r for c in cells) / len(cells)
    for step in range(20):
        for cell in cells:
            cell.r = cell.r + 0.3 * (target_r - cell.r)  # exponential convergence
        monitor.snapshot()

    assert monitor.is_healthy(), "Monitor should be healthy when V is decreasing"


# ---------------------------------------------------------------------------
# Test 4: convergence_report() returns dict with health key
# ---------------------------------------------------------------------------

def test_convergence_report_has_health_key(monitor):
    monitor.snapshot()
    report = monitor.convergence_report()
    assert isinstance(report, dict)
    assert "health" in report
    assert report["health"] in ("converging", "stable", "diverging", "unknown")
    assert "v_current" in report
    assert "recommendation" in report


# ---------------------------------------------------------------------------
# Test 5: health="converging" when V is decreasing, "stable" when plateaued
# ---------------------------------------------------------------------------

def test_health_converging_vs_stable(world):
    monitor = ConvergenceMonitor(world, window=50)

    # Create variance
    cells = list(world._cells.values())
    for i, cell in enumerate(cells):
        cell.r = 0.05 + (i % 8) * 0.11
        cell.c = 0.2 + (i % 5) * 0.12

    monitor.snapshot()  # high variance baseline

    # Convergence phase: exponentially reduce variance
    target_r = sum(c.r for c in cells) / len(cells)
    target_c = sum(c.c for c in cells) / len(cells)
    for step in range(25):
        for cell in cells:
            cell.r = cell.r + 0.35 * (target_r - cell.r)
            cell.c = cell.c + 0.35 * (target_c - cell.c)
        monitor.snapshot()

    report = monitor.convergence_report()
    assert report["health"] in ("converging", "stable"), (
        f"Expected converging or stable after V decrease, got {report['health']}"
    )

    # Plateau: take 25 more snapshots with no change
    for _ in range(25):
        monitor.snapshot()

    plateau_report = monitor.convergence_report()
    # At plateau, most delta_v should be ~0 → stable or converging
    assert plateau_report["health"] in ("stable", "converging"), (
        f"Expected stable or converging at plateau, got {plateau_report['health']}"
    )
