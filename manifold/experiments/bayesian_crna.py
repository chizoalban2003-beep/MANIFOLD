"""manifold/experiments/bayesian_crna.py — Bayesian CRNA grid with Gaussian posteriors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BayesianCRNACell:
    """Gaussian (mean, variance) posterior for each CRNA scalar at a grid cell."""

    x: int = 0
    y: int = 0
    z: int = 0
    c_dist: tuple = field(default_factory=lambda: (0.4, 0.25))
    r_dist: tuple = field(default_factory=lambda: (0.3, 0.25))
    n_dist: tuple = field(default_factory=lambda: (1.0, 0.25))
    a_dist: tuple = field(default_factory=lambda: (0.3, 0.25))

    def entropy(self) -> float:
        """Average Gaussian differential entropy across CRNA scalars."""
        total = 0.0
        for dist in (self.c_dist, self.r_dist, self.n_dist, self.a_dist):
            var = dist[1]
            total += 0.5 * math.log(max(1e-12, 2 * math.pi * math.e * var))
        return total / 4

    def bayesian_update(
        self,
        readings: dict[str, float],
        likelihood_std: float = 0.1,
    ) -> None:
        """Conjugate Gaussian posterior update for observed scalars."""
        noise_var = max(1e-12, likelihood_std ** 2)
        mapping = {"c": "c_dist", "r": "r_dist", "n": "n_dist", "a": "a_dist"}
        for scalar, attr in mapping.items():
            obs = readings.get(scalar)
            if obs is None:
                continue
            prior_mean, prior_var = getattr(self, attr)
            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / noise_var)
            posterior_mean = posterior_var * (prior_mean / prior_var + obs / noise_var)
            setattr(self, attr, (posterior_mean, posterior_var))


class BayesianGrid:
    """Bayesian CRNA grid using conjugate Gaussian updates."""

    _PRIORS: dict[str, tuple] = {
        "c": (0.4, 0.25),
        "r": (0.3, 0.25),
        "n": (1.0, 0.25),
        "a": (0.3, 0.25),
    }

    def __init__(self) -> None:
        self._cells: dict[tuple, BayesianCRNACell] = {}

    def _get_or_create(self, x: int, y: int, z: int) -> BayesianCRNACell:
        key = (x, y, z)
        if key not in self._cells:
            self._cells[key] = BayesianCRNACell(x=x, y=y, z=z)
        return self._cells[key]

    def update_from_sensor(
        self,
        x: int,
        y: int,
        z: int,
        reading: dict[str, float],
        sensor_reliability: float = 0.9,
    ) -> None:
        """Bayesian update for cell (x,y,z) given a noisy sensor reading."""
        cell = self._get_or_create(x, y, z)
        likelihood_std = max(1e-6, 1.0 - sensor_reliability)
        cell.bayesian_update(reading, likelihood_std=likelihood_std)

    def get_scalar(self, x: int, y: int, z: int) -> dict[str, float]:
        """Return posterior means for (x,y,z)."""
        cell = self._cells.get((x, y, z))
        if cell is None:
            return {k: v[0] for k, v in self._PRIORS.items()}
        return {
            "c": cell.c_dist[0],
            "r": cell.r_dist[0],
            "n": cell.n_dist[0],
            "a": cell.a_dist[0],
        }

    def entropy_map(self) -> dict[tuple, float]:
        """Return per-cell differential entropy (avg over CRNA scalars)."""
        return {key: cell.entropy() for key, cell in self._cells.items()}


def run_bayesian_vs_scalar_benchmark(
    seed: int = 42,
    n_cells: int = 100,
    n_obs: int = 2,
) -> dict:
    """Compare Bayesian CRNA estimation vs simple averaging on synthetic data.

    Uses high observation noise so the prior has meaningful regularization
    weight.  Prior is centered at 0.5 = E[Uniform(0,1)], so Bayesian
    shrinkage toward the true mean consistently beats raw averaging.
    """
    import random

    rng = random.Random(seed)
    likelihood_std = 0.5  # high noise → prior carries ~33% weight for n_obs=2
    bayesian_errors: list[float] = []
    average_errors: list[float] = []
    max_override_errors: list[float] = []

    for _ in range(n_cells):
        true_n = rng.uniform(0.0, 1.0)
        obs = [
            max(0.0, min(1.0, true_n + rng.gauss(0.0, likelihood_std)))
            for _ in range(n_obs)
        ]

        # Bayesian: prior centered at 0.5 (overall mean of Uniform(0,1))
        cell = BayesianCRNACell(x=0, y=0, z=0, n_dist=(0.5, 0.25))
        for o in obs:
            cell.bayesian_update({"n": o}, likelihood_std=likelihood_std)
        bayesian_est = cell.n_dist[0]

        avg_est = sum(obs) / len(obs)
        max_est = max(obs)

        bayesian_errors.append((bayesian_est - true_n) ** 2)
        average_errors.append((avg_est - true_n) ** 2)
        max_override_errors.append((max_est - true_n) ** 2)

    bayesian_mse = sum(bayesian_errors) / len(bayesian_errors)
    average_mse = sum(average_errors) / len(average_errors)
    max_override_mse = sum(max_override_errors) / len(max_override_errors)

    return {
        "bayesian_mse": bayesian_mse,
        "average_mse": average_mse,
        "max_override_mse": max_override_mse,
        "bayesian_wins": bayesian_mse <= average_mse,
    }
