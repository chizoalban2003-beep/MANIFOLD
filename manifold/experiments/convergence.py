"""EXP5 — NERVATURA Convergence Measurement (Lyapunov).

Empirically measures whether the NERVATURA social grid converges,
and how fast, by computing a Lyapunov candidate function V(t).
No new dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from manifold.nervatura_world import NERVATURAWorld


# ---------------------------------------------------------------------------
# ConvergenceMeasurement
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceMeasurement:
    """A single V(t) measurement."""
    timestep: int
    v_lyapunov: float       # sum of |crna - steady_state|^2
    entropy_total: float    # sum of per-cell N values (proxy for uncertainty)
    delta_v: float          # V(t) - V(t-1); negative = converging
    is_converging: bool     # True if delta_v < 0


# ---------------------------------------------------------------------------
# NERVATURAConvergenceTracker
# ---------------------------------------------------------------------------

class NERVATURAConvergenceTracker:
    """Tracks Lyapunov convergence of a NERVATURAWorld simulation."""

    def __init__(self, world: NERVATURAWorld) -> None:
        self._world = world

    def snapshot(self) -> dict:
        """Return a flat dict of all cell CRNA values keyed by (x,y,z)."""
        snap: dict = {}
        for key, cell in self._world._cells.items():
            snap[key] = {"c": cell.c, "r": cell.r, "n": cell.n, "a": cell.a}
        return snap

    def estimate_steady_state(self, snapshots: list) -> dict:
        """Estimate steady state as the mean of the last 50 snapshots.

        Returns a dict in the same format as ``snapshot()``.
        """
        if not snapshots:
            return {}

        recent = snapshots[-50:]
        # Collect all cell keys from all snapshots
        all_keys: set = set()
        for snap in recent:
            all_keys.update(snap.keys())

        steady: dict = {}
        for key in all_keys:
            values = [s[key] for s in recent if key in s]
            if not values:
                continue
            steady[key] = {
                dim: sum(v[dim] for v in values) / len(values)
                for dim in ("c", "r", "n", "a")
            }
        return steady

    def compute_lyapunov(self, current_snapshot: dict, steady_state: dict) -> float:
        """V = Σ over all cells of (C_t-C_ss)²+(R_t-R_ss)²+(N_t-N_ss)²+(A_t-A_ss)².

        Returns 0.0 when current == steady_state.
        """
        v = 0.0
        common_keys = set(current_snapshot.keys()) & set(steady_state.keys())
        for key in common_keys:
            cur = current_snapshot[key]
            ss = steady_state[key]
            v += (cur["c"] - ss["c"]) ** 2
            v += (cur["r"] - ss["r"]) ** 2
            v += (cur["n"] - ss["n"]) ** 2
            v += (cur["a"] - ss["a"]) ** 2
        return v

    def track_convergence(self, n_steps: int = 500) -> list:
        """Run the world for *n_steps*, recording V(t) at each step.

        Simulation rule per step:
        * Each simulated agent visits a random cell and reduces N by 0.05,
          harvests A (up to 0.1), and raises C slightly (x0.99).
        * The world diffuses neutrality at each step.

        Returns list of ``ConvergenceMeasurement``.
        """
        rng = random.Random(42)
        cells_list = list(self._world._cells.keys())
        if not cells_list:
            return []

        n_agents = min(4, len(cells_list))
        # Start agents at random positions
        agent_positions = [rng.choice(cells_list) for _ in range(n_agents)]

        snapshots: list = []
        measurements: list = []
        prev_v = None

        for step in range(n_steps):
            # Simulate one step of agent activity
            for idx, pos in enumerate(agent_positions):
                cell = self._world._cells.get(pos)
                if cell is not None:
                    cell.n = max(0.0, cell.n - 0.05)
                    harvested = min(cell.a, 0.1)
                    cell.a -= harvested
                    cell.c = max(0.0, cell.c * 0.99)
                # Move agent to a random neighbor (or stay)
                neighbors = [
                    (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                    if (pos[0] + dx, pos[1] + dy, pos[2] + dz) in self._world._cells
                ]
                if neighbors:
                    agent_positions[idx] = rng.choice(neighbors)

            # Diffuse neutrality (fog of war returns)
            self._world.diffuse_neutrality(decay=0.01)

            # Take snapshot and compute Lyapunov
            snap = self.snapshot()
            snapshots.append(snap)

            steady = self.estimate_steady_state(snapshots)
            v = self.compute_lyapunov(snap, steady)

            delta_v = (v - prev_v) if prev_v is not None else 0.0
            prev_v = v

            entropy_total = sum(s["n"] for s in snap.values())

            measurements.append(ConvergenceMeasurement(
                timestep=step,
                v_lyapunov=round(v, 8),
                entropy_total=round(entropy_total, 4),
                delta_v=round(delta_v, 8),
                is_converging=delta_v < 0.0,
            ))

        return measurements

    def convergence_report(self, measurements: list) -> dict:
        """Summarise convergence from a measurement sequence.

        Returns
        -------
        dict with keys: converges, convergence_step, final_v, initial_v,
            monotone_ratio, rate.
        """
        if not measurements:
            return {
                "converges": False,
                "convergence_step": -1,
                "final_v": 0.0,
                "initial_v": 0.0,
                "monotone_ratio": 0.0,
                "rate": 0.0,
            }

        initial_v = measurements[0].v_lyapunov
        final_v = measurements[-1].v_lyapunov
        converges = final_v < initial_v

        # Step where delta_v is negative for the last time
        convergence_step = -1
        for m in measurements:
            if m.is_converging:
                convergence_step = m.timestep

        n = len(measurements)
        monotone_count = sum(1 for m in measurements if m.is_converging)
        monotone_ratio = monotone_count / n if n > 0 else 0.0

        # Estimate exponential decay rate: V(t) ≈ V0 * exp(-rate * t)
        rate = 0.0
        if initial_v > 1e-9 and final_v < initial_v:
            try:
                rate = -math.log(max(final_v / initial_v, 1e-9)) / max(n - 1, 1)
            except (ValueError, ZeroDivisionError):
                rate = 0.0

        return {
            "converges": converges,
            "convergence_step": convergence_step,
            "final_v": round(final_v, 8),
            "initial_v": round(initial_v, 8),
            "monotone_ratio": round(monotone_ratio, 4),
            "rate": round(rate, 8),
        }


# ---------------------------------------------------------------------------
# Benchmark entry point (used by /experiments/convergence)
# ---------------------------------------------------------------------------

def run_convergence_benchmark() -> dict:
    """Instantiate a small world, run convergence tracking, return report."""
    world = NERVATURAWorld(5, 5, 1, default_crna=(0.5, 0.4, 1.0, 0.5))
    tracker = NERVATURAConvergenceTracker(world)
    measurements = tracker.track_convergence(n_steps=200)
    report = tracker.convergence_report(measurements)
    report["measurements_count"] = len(measurements)
    return report
