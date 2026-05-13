"""Tests for EXP5 — NERVATURA Convergence Measurement (Lyapunov)."""
from __future__ import annotations

from manifold.nervatura_world import NERVATURAWorld
from manifold.experiments.convergence import (
    ConvergenceMeasurement,
    NERVATURAConvergenceTracker,
    run_convergence_benchmark,
)


def _tracker() -> NERVATURAConvergenceTracker:
    world = NERVATURAWorld(4, 4, 1, default_crna=(0.5, 0.4, 1.0, 0.5))
    return NERVATURAConvergenceTracker(world)


# Test 1: NERVATURAConvergenceTracker.snapshot returns dict with cell keys
def test_snapshot_returns_cell_dict():
    tracker = _tracker()
    snap = tracker.snapshot()
    assert isinstance(snap, dict)
    assert len(snap) > 0
    # Each key should be a (x,y,z) tuple, each value should have c,r,n,a
    key = next(iter(snap))
    assert isinstance(key, tuple) and len(key) == 3
    value = snap[key]
    for dim in ("c", "r", "n", "a"):
        assert dim in value, f"Missing key {dim!r} in snapshot cell"


# Test 2: compute_lyapunov returns 0 when current == steady_state
def test_lyapunov_zero_for_equal_states():
    tracker = _tracker()
    snap = tracker.snapshot()
    v = tracker.compute_lyapunov(snap, snap)
    assert v == 0.0 or abs(v) < 1e-10, f"V should be 0 when states equal, got {v}"


# Test 3: compute_lyapunov returns positive value when states differ
def test_lyapunov_positive_for_different_states():
    tracker = _tracker()
    current = tracker.snapshot()
    # Create a different steady state
    steady = {k: {"c": v["c"] + 0.1, "r": v["r"], "n": v["n"], "a": v["a"]} for k, v in current.items()}
    v = tracker.compute_lyapunov(current, steady)
    assert v > 0.0, f"V should be positive when states differ, got {v}"


# Test 4: track_convergence returns list of ConvergenceMeasurement
def test_track_convergence_returns_measurements():
    tracker = _tracker()
    measurements = tracker.track_convergence(n_steps=10)
    assert isinstance(measurements, list)
    assert len(measurements) == 10
    for m in measurements:
        assert isinstance(m, ConvergenceMeasurement)
        assert isinstance(m.v_lyapunov, float)
        assert isinstance(m.timestep, int)
        assert isinstance(m.is_converging, bool)


# Test 5: convergence_report returns dict with converges key
def test_convergence_report_has_converges_key():
    tracker = _tracker()
    measurements = tracker.track_convergence(n_steps=20)
    report = tracker.convergence_report(measurements)
    assert isinstance(report, dict)
    assert "converges" in report
    assert "final_v" in report
    assert "initial_v" in report
    assert "monotone_ratio" in report
    assert "rate" in report
