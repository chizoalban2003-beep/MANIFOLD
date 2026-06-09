"""manifold/nervatura_world.py — 3D voxel grid of CRNA cells."""

from __future__ import annotations

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


class NERVATURAWorld:
    """3D voxel grid where each cell carries CRNA values.

    Parameters
    ----------
    width, depth, height:
        Grid dimensions.
    default_crna:
        Tuple ``(c, r, n, a)`` for all cells at construction time.
    """

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

    def set_cell(self, x: int, y: int, z: int, **kwargs: float) -> None:
        cell = self._cells.get((x, y, z))
        if cell is None:
            return
        for attr, val in kwargs.items():
            setattr(cell, attr, val)
