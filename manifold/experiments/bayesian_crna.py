"""manifold/experiments/bayesian_crna.py — Bayesian CRNA grid with Gaussian posteriors."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BayesianCRNACell:
    """Gaussian (mean, variance) for each CRNA scalar at a grid cell."""

    c_mean: float = 0.4
    c_var: float = 0.25
    r_mean: float = 0.3
    r_var: float = 0.25
    n_mean: float = 1.0
    n_var: float = 0.25
    a_mean: float = 0.3
    a_var: float = 0.25


class BayesianGrid:
    """Bayesian CRNA grid using conjugate Gaussian updates.

    Each cell maintains a posterior mean and variance per scalar.
    Prior: mean = prior_mean, variance = prior_var (flat/uninformed).
    """

    _PRIORS: dict[str, tuple[float, float]] = {
        "c": (0.4, 0.25),
        "r": (0.3, 0.25),
        "n": (1.0, 0.25),
        "a": (0.3, 0.25),
    }

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int, int], BayesianCRNACell] = {}

    # ------------------------------------------------------------------

    def _get_or_create(self, x: int, y: int, z: int) -> BayesianCRNACell:
        key = (x, y, z)
        if key not in self._cells:
            self._cells[key] = BayesianCRNACell()
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
        noise_var = max(1e-6, (1.0 - sensor_reliability) ** 2)
        for scalar in ("c", "r", "n", "a"):
            obs = reading.get(scalar)
            if obs is None:
                continue
            prior_mean = getattr(cell, f"{scalar}_mean")
            prior_var = getattr(cell, f"{scalar}_var")
            # Gaussian conjugate update
            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / noise_var)
            posterior_mean = posterior_var * (prior_mean / prior_var + obs / noise_var)
            setattr(cell, f"{scalar}_mean", posterior_mean)
            setattr(cell, f"{scalar}_var", posterior_var)

    def get_scalar(self, x: int, y: int, z: int) -> dict[str, float]:
        """Return posterior means for (x,y,z)."""
        cell = self._cells.get((x, y, z))
        if cell is None:
            return {k: v[0] for k, v in self._PRIORS.items()}
        return {
            "c": cell.c_mean,
            "r": cell.r_mean,
            "n": cell.n_mean,
            "a": cell.a_mean,
        }

    def entropy_map(self) -> dict[tuple[int, int, int], float]:
        """Return per-cell differential entropy (avg over CRNA scalars)."""
        result: dict[tuple[int, int, int], float] = {}
        for key, cell in self._cells.items():
            entropies = []
            for scalar in ("c", "r", "n", "a"):
                var = getattr(cell, f"{scalar}_var")
                # Gaussian differential entropy: 0.5 * ln(2*pi*e*var)
                entropies.append(0.5 * math.log(max(1e-12, 2 * math.pi * math.e * var)))
            result[key] = sum(entropies) / len(entropies)
        return result
