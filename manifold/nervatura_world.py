"""manifold/nervatura_world.py — 3D voxel grid of CRNA cells."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class NERVATURACell:
    """A single cell in the NERVATURA world grid."""

    x: int
    y: int
    z: int
    c: float = 0.4  # Cost
    r: float = 0.3  # Risk
    n: float = 1.0  # Neutrality
    a: float = 0.3  # Asset
    domain: str = ""
    age: int = 0

    def reduce_neutrality(self, amount: float) -> None:
        self.n = max(0.0, self.n - amount)
        self.age += 1

    def terraform(self, cost_reduction: float) -> None:
        self.c = max(0.0, self.c - cost_reduction)

    def harvest(self, amount: float) -> float:
        got = min(self.a, amount)
        self.a -= got
        return got


class NERVATURAWorld:
    """3D voxel grid where each cell carries CRNA values."""

    def __init__(
        self,
        width: int,
        depth: int,
        height: int,
        default_crna: tuple[float, float, float, float] = (0.4, 0.3, 1.0, 0.3),
    ) -> None:
        self.width = width
        self.depth = depth
        self.height = height
        c, r, n, a = default_crna
        self._cells: dict[tuple[int, int, int], NERVATURACell] = {}
        for x in range(width):
            for y in range(depth):
                for z in range(height):
                    self._cells[(x, y, z)] = NERVATURACell(x=x, y=y, z=z, c=c, r=r, n=n, a=a)

    def cell(self, x: int, y: int, z: int) -> NERVATURACell | None:
        return self._cells.get((x, y, z))

    def set_cell(self, x: int, y: int, z: int, **kwargs) -> None:
        cell = self._cells.get((x, y, z))
        if cell is None:
            return
        for attr, val in kwargs.items():
            setattr(cell, attr, val)

    def neighbours(self, x: int, y: int, z: int) -> list[NERVATURACell]:
        result = []
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nb = self._cells.get((x+dx, y+dy, z+dz))
            if nb is not None:
                result.append(nb)
        return result

    def summary(self) -> dict:
        return {"total_cells": len(self._cells)}

    def diffuse_neutrality(self, decay: float = 0.01) -> None:
        for cell in self._cells.values():
            cell.n = min(1.0, cell.n + decay)

    def to_json(self) -> str:
        data = {
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "cells": [
                {
                    "x": c.x, "y": c.y, "z": c.z,
                    "c_val": c.c, "r_val": c.r, "n_val": c.n, "a_val": c.a,
                    "domain": c.domain, "age": c.age,
                }
                for c in self._cells.values()
            ],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "NERVATURAWorld":
        d = json.loads(data)
        world = cls(width=d["width"], depth=d["depth"], height=d["height"])
        for cd in d["cells"]:
            key = (cd["x"], cd["y"], cd["z"])
            if key in world._cells:
                cell = world._cells[key]
                cell.c = cd["c_val"]
                cell.r = cd["r_val"]
                cell.n = cd["n_val"]
                cell.a = cd["a_val"]
                cell.domain = cd.get("domain", "")
                cell.age = cd.get("age", 0)
        return world
