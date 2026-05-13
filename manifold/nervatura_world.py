"""NERVATURAWorld — 3-D CRNA voxel grid engine.

Represents any problem domain (physical or digital) as navigable
terrain.  Mathematical core for MANIFOLD Physical and digital problem
mapping.  Zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional


# Default categorical belief over cell states (sums to 1.0)
_DEFAULT_BELIEF = {
    "empty": 0.6,
    "agent": 0.05,
    "obstacle": 0.1,
    "hazardous": 0.05,
    "unknown": 0.2,
}
_NUM_STATES = len(_DEFAULT_BELIEF)


@dataclass
class NERVATURACell:
    x: int
    y: int
    z: int
    c: float = 0.5   # Cost to traverse
    r: float = 0.5   # Risk of failure
    n: float = 1.0   # Neutrality (1.0 = completely unknown; now backed by Shannon H)
    a: float = 0.0   # Asset available
    age: int = 0
    last_visited: float = 0.0
    domain: str = "general"
    # Categorical belief distribution for formal Shannon entropy N
    belief: dict = field(default_factory=lambda: dict(_DEFAULT_BELIEF))

    # ------------------------------------------------------------------
    # Formal Shannon entropy Neutrality (PROMPT A1 — closes Gap A)
    # ------------------------------------------------------------------

    def update_belief(self, observation: str, likelihood: float = 0.85) -> None:
        """Bayesian update of *belief* given an observation.

        Parameters
        ----------
        observation:
            The observed state (key in *belief*).  One of: "empty", "agent",
            "obstacle", "hazardous", "unknown".
        likelihood:
            Sensor reliability: P(obs | observed_state).  Default 0.85.
        """
        if observation not in self.belief:
            return  # unknown observation — silently ignore
        num_states = len(self.belief)
        other_likelihood = (1.0 - likelihood) / max(num_states - 1, 1)
        posterior: dict = {}
        for state, prior_p in self.belief.items():
            lk = likelihood if state == observation else other_likelihood
            posterior[state] = prior_p * lk
        total = sum(posterior.values())
        if total <= 0.0:
            return  # degenerate — keep current belief
        self.belief = {k: v / total for k, v in posterior.items()}

    def formal_n(self) -> float:
        """Return normalised Shannon entropy of *belief* in [0, 1].

        H = -Σ p·log(p)  /  log(|belief|)

        Returns 1.0 when all states are equally likely (maximum uncertainty),
        0.0 when exactly one state has probability 1 (fully known cell).
        """
        num_states = len(self.belief)
        if num_states <= 1:
            return 0.0
        h = -sum(p * math.log(p) for p in self.belief.values() if p > 0)
        return h / math.log(num_states)

    def sync_n(self) -> None:
        """Set self.n = formal_n().  Call after every update_belief()."""
        self.n = self.formal_n()

    # ------------------------------------------------------------------
    # Original NERVATURA methods
    # ------------------------------------------------------------------

    def traversal_cost(self) -> float:
        """NERVATURA cost function: c + r * 0.3."""
        return self.c + self.r * 0.3

    def is_navigable(self, risk_budget: float) -> bool:
        return self.r < risk_budget and self.c < 0.99

    def reduce_neutrality(self, amount: float = 0.3) -> None:
        """Reduce fog of war — cell becomes more known."""
        self.n = max(0.0, self.n - amount)
        self.age += 1
        self.last_visited = time.time()

    def terraform(self, cost_reduction: float = 0.15) -> None:
        """Terraforming: reduce traversal cost."""
        self.c = max(0.0, self.c * (1 - cost_reduction))

    def harvest(self, amount: float = 0.5) -> float:
        """Harvest asset from this cell.  Returns amount actually harvested."""
        harvested = min(self.a, amount)
        self.a -= harvested
        return harvested


class NERVATURAWorld:
    """3-D grid of NERVATURACell objects representing any domain as terrain."""

    def __init__(
        self,
        width: int,
        depth: int,
        height: int,
        default_crna: tuple = (0.5, 0.5, 1.0, 0.0),
    ) -> None:
        self.width = width
        self.depth = depth
        self.height = height
        self._default_crna = default_crna
        self._cells: dict[tuple[int, int, int], NERVATURACell] = {}
        c, r, n, a = default_crna
        for x in range(width):
            for y in range(depth):
                for z in range(height):
                    self._cells[(x, y, z)] = NERVATURACell(x=x, y=y, z=z, c=c, r=r, n=n, a=a)

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def cell(self, x: int, y: int, z: int) -> Optional[NERVATURACell]:
        return self._cells.get((x, y, z))

    def set_cell(
        self,
        x: int,
        y: int,
        z: int,
        c: float,
        r: float,
        n: float,
        a: float,
        domain: str = "general",
    ) -> None:
        key = (x, y, z)
        if key in self._cells:
            cell = self._cells[key]
            cell.c = c
            cell.r = r
            cell.n = n
            cell.a = a
            cell.domain = domain
        else:
            cell = NERVATURACell(x=x, y=y, z=z, c=c, r=r, n=n, a=a, domain=domain)
            self._cells[key] = cell
        # Sync formal Shannon entropy N after explicit set
        cell.sync_n()

    def neighbours(self, x: int, y: int, z: int) -> list[NERVATURACell]:
        """Return 6 face-adjacent navigable cells."""
        result = []
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nb = self._cells.get((x + dx, y + dy, z + dz))
            if nb is not None:
                result.append(nb)
        return result

    # ------------------------------------------------------------------
    # World mechanics
    # ------------------------------------------------------------------

    def diffuse_neutrality(self, decay: float = 0.05) -> None:
        """Fog of war returns over time: n += decay for all cells with n < 1.0."""
        for cell in self._cells.values():
            if cell.n < 1.0:
                cell.n = min(1.0, cell.n + decay)

    def apply_bus_updates(self) -> None:
        """Read from DynamicGrid and update matching cells."""
        try:
            from .dynamic_grid import get_grid
            grid = get_grid()
            for (x, y, z), vals in grid.all_cells().items():
                if (x, y, z) in self._cells:
                    cell = self._cells[(x, y, z)]
                    cell.c = vals.c
                    cell.r = vals.r
                    cell.n = vals.n
                    cell.a = vals.a
        except Exception as exc:  # noqa: BLE001
            logging.debug("apply_bus_updates: grid sync failed: %s", exc)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return world statistics."""
        cells = list(self._cells.values())
        total = len(cells)
        if total == 0:
            return {"total_cells": 0}
        avg_c = sum(c.c for c in cells) / total
        avg_r = sum(c.r for c in cells) / total
        avg_n = sum(c.n for c in cells) / total
        avg_a = sum(c.a for c in cells) / total
        fully_explored = sum(1 for c in cells if c.n < 0.1)
        unknown = sum(1 for c in cells if c.n > 0.9)
        return {
            "total_cells": total,
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "avg_c": round(avg_c, 4),
            "avg_r": round(avg_r, 4),
            "avg_n": round(avg_n, 4),
            "avg_a": round(avg_a, 4),
            "fully_explored": fully_explored,
            "unknown": unknown,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise world to JSON."""
        data = {
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "default_crna": list(self._default_crna),
            "cells": [
                {
                    "x": cell.x, "y": cell.y, "z": cell.z,
                    "c": cell.c, "r": cell.r, "n": cell.n, "a": cell.a,
                    "age": cell.age, "last_visited": cell.last_visited,
                    "domain": cell.domain,
                }
                for cell in self._cells.values()
            ],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "NERVATURAWorld":
        """Deserialise from JSON.  Returns a new NERVATURAWorld instance."""
        d = json.loads(data)
        world = cls(
            width=d["width"],
            depth=d["depth"],
            height=d["height"],
            default_crna=tuple(d.get("default_crna", [0.5, 0.5, 1.0, 0.0])),
        )
        for cd in d.get("cells", []):
            key = (cd["x"], cd["y"], cd["z"])
            if key in world._cells:
                cell = world._cells[key]
            else:
                cell = NERVATURACell(x=cd["x"], y=cd["y"], z=cd["z"])
                world._cells[key] = cell
            cell.c = cd["c"]
            cell.r = cd["r"]
            cell.n = cd["n"]
            cell.a = cd["a"]
            cell.age = cd.get("age", 0)
            cell.last_visited = cd.get("last_visited", 0.0)
            cell.domain = cd.get("domain", "general")
        return world
