"""EXP5 — NERVATURA Convergence Measurement (Lyapunov).

Empirically measures whether the NERVATURA social grid converges,
and how fast, by computing a Lyapunov candidate function V(t).
No new dependencies.

Fix (8.4): Enhanced simulation model with:
  - Heterogeneous agent archetypes (Explorer, Harvester, Terraformer, Defender)
  - Resource replenishment (A slowly restores after harvest)
  - Non-stationary hazards (R spikes probabilistically)
  - Agent strategy adaptation (agents switch archetype based on local cell state)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

from manifold.nervatura_world import NERVATURAWorld


# ---------------------------------------------------------------------------
# Agent archetypes
# ---------------------------------------------------------------------------

class AgentArchetype(Enum):
    EXPLORER = "explorer"       # Reduces N — scout/mapping
    HARVESTER = "harvester"     # Reduces A — resource collection
    TERRAFORMER = "terraformer" # Reduces C — cost reduction
    DEFENDER = "defender"       # Reduces R — hazard mitigation


# ---------------------------------------------------------------------------
# ConvergenceMeasurement
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceMeasurement:
    """A single V(t) measurement."""
    timestep: int
    v_lyapunov: float
    entropy_total: float
    delta_v: float
    is_converging: bool
    archetype_counts: dict  # {archetype_name: count} — snapshot of agent composition


# ---------------------------------------------------------------------------
# NERVATURAConvergenceTracker
# ---------------------------------------------------------------------------

class NERVATURAConvergenceTracker:
    """Tracks Lyapunov convergence of a NERVATURAWorld simulation.

    Uses a heterogeneous agent model where agents adapt their behavior
    (archetype) based on the local cell state, and the environment
    exhibits non-stationary hazard events and resource replenishment.
    """

    # Archetype switching thresholds
    _SWITCH_THRESHOLDS = {
        # Switch to EXPLORER when N > 0.7 (cell poorly mapped)
        AgentArchetype.EXPLORER: ("n", ">", 0.7),
        # Switch to HARVESTER when A > 0.6 (rich resource deposit)
        AgentArchetype.HARVESTER: ("a", ">", 0.6),
        # Switch to DEFENDER when R > 0.6 (dangerous cell)
        AgentArchetype.DEFENDER: ("r", ">", 0.6),
        # Otherwise TERRAFORMER (reduce cost)
        AgentArchetype.TERRAFORMER: ("c", ">", 0.0),
    }

    def __init__(self, world: NERVATURAWorld) -> None:
        self._world = world

    def snapshot(self) -> dict:
        snap: dict = {}
        for key, cell in self._world._cells.items():
            snap[key] = {"c": cell.c, "r": cell.r, "n": cell.n, "a": cell.a}
        return snap

    def estimate_steady_state(self, snapshots: list) -> dict:
        """Estimate steady state as the mean of the last 50 snapshots."""
        if not snapshots:
            return {}
        recent = snapshots[-50:]
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
        """V = Σ over all cells of (C_t-C_ss)²+(R_t-R_ss)²+(N_t-N_ss)²+(A_t-A_ss)²."""
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

    @staticmethod
    def _select_archetype(cell) -> AgentArchetype:
        """Select agent archetype based on cell state (adaptive strategy)."""
        if cell.n > 0.7:
            return AgentArchetype.EXPLORER
        if cell.r > 0.6:
            return AgentArchetype.DEFENDER
        if cell.a > 0.6:
            return AgentArchetype.HARVESTER
        return AgentArchetype.TERRAFORMER

    @staticmethod
    def _apply_archetype(cell, archetype: AgentArchetype) -> None:
        """Apply archetype-specific cell modification."""
        if archetype == AgentArchetype.EXPLORER:
            cell.n = max(0.0, cell.n - 0.08)
        elif archetype == AgentArchetype.HARVESTER:
            harvested = min(cell.a, 0.12)
            cell.a -= harvested
        elif archetype == AgentArchetype.TERRAFORMER:
            cell.c = max(0.0, cell.c * 0.97)
        elif archetype == AgentArchetype.DEFENDER:
            cell.r = max(0.0, cell.r * 0.92)

    def track_convergence(self, n_steps: int = 500) -> list:
        """Run the world for *n_steps* with heterogeneous adaptive agents.

        Per-step dynamics:
        - Each agent selects its archetype based on current cell state.
        - Agent applies archetype-specific effect on the cell.
        - Agent moves to a random navigable neighbor.
        - World-level: neutrality diffusion, resource replenishment (A +=
          replenish_rate for cells below 0.8), probabilistic hazard spikes
          (R +=0.2 for a random 3% of cells — non-stationary disturbance).

        Returns list of ``ConvergenceMeasurement``.
        """
        rng = random.Random(42)
        cells_list = list(self._world._cells.keys())
        if not cells_list:
            return []

        n_agents = min(6, len(cells_list))
        agent_positions = [rng.choice(cells_list) for _ in range(n_agents)]
        agent_archetypes = [AgentArchetype.EXPLORER] * n_agents

        snapshots: list = []
        measurements: list = []
        prev_v = None

        # Replenishment rate per step (A slowly restores in unvisited cells)
        replenish_rate = 0.005
        # Hazard spike probability per cell per step
        hazard_prob = 0.03

        for step in range(n_steps):
            archetype_counts: dict[str, int] = {a.value: 0 for a in AgentArchetype}

            # Agent activity
            for idx, pos in enumerate(agent_positions):
                cell = self._world._cells.get(pos)
                if cell is not None:
                    archetype = self._select_archetype(cell)
                    agent_archetypes[idx] = archetype
                    archetype_counts[archetype.value] += 1
                    self._apply_archetype(cell, archetype)

                neighbors = [
                    (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    for dx, dy, dz in [
                        (1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1),
                    ]
                    if (pos[0] + dx, pos[1] + dy, pos[2] + dz) in self._world._cells
                ]
                if neighbors:
                    agent_positions[idx] = rng.choice(neighbors)

            # World dynamics: neutrality diffusion
            self._world.diffuse_neutrality(decay=0.01)

            # World dynamics: resource replenishment
            for cell in self._world._cells.values():
                if cell.a < 0.8:
                    cell.a = min(0.8, cell.a + replenish_rate)

            # World dynamics: non-stationary hazard spikes
            spike_count = max(1, int(len(cells_list) * hazard_prob))
            for _ in range(spike_count):
                spike_pos = rng.choice(cells_list)
                spike_cell = self._world._cells.get(spike_pos)
                if spike_cell is not None:
                    spike_cell.r = min(1.0, spike_cell.r + 0.2)

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
                archetype_counts=dict(archetype_counts),
            ))

        return measurements

    def convergence_report(self, measurements: list) -> dict:
        """Summarise convergence from a measurement sequence."""
        if not measurements:
            return {
                "converges": False,
                "convergence_step": -1,
                "final_v": 0.0,
                "initial_v": 0.0,
                "monotone_ratio": 0.0,
                "rate": 0.0,
                "archetype_distribution": {},
            }

        initial_v = measurements[0].v_lyapunov
        final_v = measurements[-1].v_lyapunov
        converges = final_v < initial_v

        convergence_step = -1
        for m in measurements:
            if m.is_converging:
                convergence_step = m.timestep

        n = len(measurements)
        monotone_count = sum(1 for m in measurements if m.is_converging)
        monotone_ratio = monotone_count / n if n > 0 else 0.0

        rate = 0.0
        if initial_v > 1e-9 and final_v < initial_v:
            try:
                rate = -math.log(max(final_v / initial_v, 1e-9)) / max(n - 1, 1)
            except (ValueError, ZeroDivisionError):
                rate = 0.0

        # Aggregate archetype distribution across all steps
        agg: dict[str, int] = {}
        for m in measurements:
            for k, v in m.archetype_counts.items():
                agg[k] = agg.get(k, 0) + v
        total_actions = sum(agg.values()) or 1
        archetype_dist = {k: round(v / total_actions, 4) for k, v in agg.items()}

        return {
            "converges": converges,
            "convergence_step": convergence_step,
            "final_v": round(final_v, 8),
            "initial_v": round(initial_v, 8),
            "monotone_ratio": round(monotone_ratio, 4),
            "rate": round(rate, 8),
            "archetype_distribution": archetype_dist,
        }


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

def run_convergence_benchmark() -> dict:
    """Instantiate a small world, run convergence tracking, return report.

    Also runs the original linear-decay baseline for comparison.
    """
    world = NERVATURAWorld(5, 5, 1, default_crna=(0.5, 0.4, 1.0, 0.5))
    tracker = NERVATURAConvergenceTracker(world)
    measurements = tracker.track_convergence(n_steps=200)
    report = tracker.convergence_report(measurements)
    report["measurements_count"] = len(measurements)
    return report
