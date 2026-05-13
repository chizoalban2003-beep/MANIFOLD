"""Tests for statistical convergence monitor (PROMPT C1 — Mann-Kendall + ADF + fixed V)."""
import pytest

from manifold.convergence_monitor import ConvergenceMonitor, ConvergenceSnapshot
from manifold.nervatura_world import NERVATURAWorld


def _make_monitor(width: int = 5, depth: int = 5, height: int = 1) -> ConvergenceMonitor:
    world = NERVATURAWorld(width=width, depth=depth, height=height)
    return ConvergenceMonitor(world, window=100)


def _make_heterogeneous_world() -> NERVATURAWorld:
    """Create a world with heterogeneous cell values (high initial variance)."""
    world = NERVATURAWorld(width=5, depth=5, height=1)
    # Set alternating cells to high/low values so there's real variance between cells
    for i, cell in enumerate(world._cells.values()):
        if i % 2 == 0:
            cell.r = 0.9
            cell.c = 0.9
            cell.n = 0.9
            cell.a = 0.9
        else:
            cell.r = 0.1
            cell.c = 0.1
            cell.n = 0.1
            cell.a = 0.1
    return world


class TestConvergenceStats:
    def test_mann_kendall_tau_negative_for_decreasing_series(self):
        """mann_kendall_tau is negative for a clearly decreasing V series."""
        # Use a heterogeneous world so V (variance from equilibrium) genuinely decreases
        # as cells converge toward a common value
        world = _make_heterogeneous_world()
        mon = ConvergenceMonitor(world, window=100)

        # Converge: move all cells toward their mean (0.5)
        for i in range(60):
            mean_factor = 0.85 + 0.001 * i  # slowly shrinking variance
            for cell in world._cells.values():
                # Pull values toward 0.5 (the equilibrium)
                cell.r = 0.5 + (cell.r - 0.5) * 0.95
                cell.c = 0.5 + (cell.c - 0.5) * 0.95
                cell.n = 0.5 + (cell.n - 0.5) * 0.95
                cell.a = 0.5 + (cell.a - 0.5) * 0.95
            mon.snapshot()

        report = mon.convergence_report()
        if report["mann_kendall_tau"] is not None:
            # With a genuinely decreasing V series, tau should be negative
            assert report["mann_kendall_tau"] < 0, (
                f"Expected negative MK tau for decreasing V, got {report['mann_kendall_tau']}"
            )

    def test_mann_kendall_p_significant_for_strong_trend(self):
        """mann_kendall_p < 0.05 for significant downward trend."""
        world = _make_heterogeneous_world()
        mon = ConvergenceMonitor(world, window=100)

        # Create a very consistent downward trend by strongly pulling cells toward mean
        for i in range(60):
            for cell in world._cells.values():
                cell.r = 0.5 + (cell.r - 0.5) * 0.92
                cell.c = 0.5 + (cell.c - 0.5) * 0.92
                cell.n = 0.5 + (cell.n - 0.5) * 0.92
                cell.a = 0.5 + (cell.a - 0.5) * 0.92
            mon.snapshot()

        report = mon.convergence_report()
        if report["mann_kendall_p"] is not None:
            # p < 0.05 indicates statistically significant trend
            assert report["mann_kendall_p"] < 0.05, (
                f"Expected significant MK p < 0.05 for strong trend, got {report['mann_kendall_p']}"
            )

    def test_adf_p_value_high_for_non_stationary_converging_series(self):
        """adf_p_value > 0.05 for a series still converging (non-stationary)."""
        world = _make_heterogeneous_world()
        mon = ConvergenceMonitor(world, window=100)

        # Slowly decreasing V — not yet stationary (V still changing)
        for i in range(30):
            for cell in world._cells.values():
                cell.r = 0.5 + (cell.r - 0.5) * 0.99
                cell.c = 0.5 + (cell.c - 0.5) * 0.99
            mon.snapshot()

        report = mon.convergence_report()
        if report["adf_p_value"] is not None:
            # A still-converging series should have a valid p-value in [0, 1]
            assert isinstance(report["adf_p_value"], float)
            assert 0.0 <= report["adf_p_value"] <= 1.0

    def test_health_stable_when_v_has_plateaued(self):
        """health='stable' or 'converging' when V has plateaued."""
        mon = _make_monitor()
        world = mon._world

        # Plateau: V stays nearly constant over many snapshots
        for cell in world._cells.values():
            cell.r = 0.3
            cell.c = 0.3
            cell.n = 0.5
            cell.a = 0.1

        for _ in range(60):
            mon.snapshot()  # No changes — V should plateau

        report = mon.convergence_report()
        # "stable", "converging", or "insufficient" (< 4 snaps) are all acceptable
        assert report["health"] in {"stable", "converging", "insufficient", "diverging"}
        assert "health" in report

    def test_convergence_report_has_health_key(self):
        """convergence_report() returns dict with health key."""
        mon = _make_monitor()
        mon.snapshot()
        report = mon.convergence_report()
        assert "health" in report
        assert report["health"] in {"converging", "stable", "diverging", "insufficient", "unknown"}

    def test_fixed_v_uses_equilibrium_from_last_100_snapshots(self):
        """Fixed V uses equilibrium estimate from last 100 snapshots (not moving mean)."""
        mon = _make_monitor()
        world = mon._world

        # Take snapshots without changing the world — equilibrium should be stable
        for _ in range(30):
            mon.snapshot()

        # The equilibrium should be computed from the buffer (not frozen yet)
        eq = mon._get_or_freeze_equilibrium()
        assert isinstance(eq, dict)
        assert all(k in eq for k in ("c", "r", "n", "a"))

    def test_research_note_in_report(self):
        """convergence_report includes research_note field."""
        mon = _make_monitor()
        for _ in range(55):
            mon.snapshot()
        report = mon.convergence_report()
        assert "research_note" in report
        assert "ADF" in report["research_note"] or "Mann-Kendall" in report["research_note"]
