"""SpaceIngestion — convert physical floor plan JSON to CRNA grid.

Populates DynamicGrid with base CRNA values for every cell described
in the floor plan.  Zero external dependencies.

Floor plan JSON format::

    {
      "name": "Home",
      "width": 10, "depth": 10, "height": 3,
      "rooms": [
        {
          "name": "Kitchen",
          "bounds": {"x":[0,4], "y":[0,4], "z":[0,3]},
          "crna": {"c":0.3, "r":0.6, "n":0.2, "a":0.8},
          "tags": ["high_risk_appliances"]
        }
      ],
      "obstacles": [
        {"type": "wall",
         "bounds": {"x":[4,5], "y":[0,10], "z":[0,3]},
         "crna": {"c":1.0, "r":0.0, "n":0.0, "a":0.0}}
      ],
      "paths": [
        {"name": "Main corridor",
         "bounds": {"x":[4,6], "y":[0,10], "z":[0,1]},
         "crna": {"c":0.1, "r":0.1, "n":0.05, "a":0.0}}
      ]
    }
"""

from __future__ import annotations

import json
import sys
import os

# Allow import from the repo root when running as a standalone package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manifold.dynamic_grid import get_grid


class SpaceIngestion:
    """Converts physical floor plan descriptions to CRNA grid cells."""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_floorplan(self, json_path: str) -> dict:
        """Load and validate a floor plan JSON file."""
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._validate(data)
        return data

    def _validate(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("Floor plan must be a JSON object")
        if "rooms" not in data and "obstacles" not in data and "paths" not in data:
            raise ValueError("Floor plan must have at least one of: rooms, obstacles, paths")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, floorplan: dict) -> int:
        """Populate DynamicGrid with every cell in the floor plan.

        Returns the total number of cells populated.
        """
        grid = get_grid()
        count = 0
        for section_key in ("rooms", "obstacles", "paths"):
            for item in floorplan.get(section_key, []):
                count += self._ingest_item(grid, item)
        return count

    def _ingest_item(self, grid, item: dict) -> int:
        bounds = item.get("bounds", {})
        crna = item.get("crna", {})
        c = float(crna.get("c", 0.5))
        r = float(crna.get("r", 0.5))
        n = float(crna.get("n", 0.5))
        a = float(crna.get("a", 0.0))
        x_range = bounds.get("x", [0, 1])
        y_range = bounds.get("y", [0, 1])
        z_range = bounds.get("z", [0, 1])
        count = 0
        for x in range(int(x_range[0]), int(x_range[1])):
            for y in range(int(y_range[0]), int(y_range[1])):
                for z in range(int(z_range[0]), int(z_range[1])):
                    grid.set_base(x, y, z, c=c, r=r, n=n, a=a)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Post-ingestion policy
    # ------------------------------------------------------------------

    def apply_room_policy(self, room_name: str, policy: dict, floorplan: dict) -> int:
        """Apply a policy override to all cells in a named room.

        Parameters
        ----------
        room_name:
            Room name to match (case-insensitive).
        policy:
            Dict with optional keys: ``r_override``, ``c_override``,
            ``n_override``, ``a_override``, ``ttl``, ``reason``.
        floorplan:
            The loaded floor plan dict (to look up room bounds).
        """
        grid = get_grid()
        ttl = float(policy.get("ttl", 3600.0))
        reason = str(policy.get("reason", "policy_override"))

        count = 0
        for room in floorplan.get("rooms", []):
            if room.get("name", "").lower() != room_name.lower():
                continue
            bounds = room.get("bounds", {})
            x_range = bounds.get("x", [0, 1])
            y_range = bounds.get("y", [0, 1])
            z_range = bounds.get("z", [0, 1])
            for x in range(int(x_range[0]), int(x_range[1])):
                for y in range(int(y_range[0]), int(y_range[1])):
                    for z in range(int(z_range[0]), int(z_range[1])):
                        current = grid.get(x, y, z)
                        ov_r = float(policy.get("r_override", current.r))
                        ov_c = float(policy.get("c_override", current.c))
                        ov_n = float(policy.get("n_override", current.n))
                        ov_a = float(policy.get("a_override", current.a))
                        key = (x, y, z)
                        if key not in grid._cells:
                            grid.set_base(x, y, z, c=current.c, r=current.r,
                                          n=current.n, a=current.a)
                        grid._cells[key].add_override(
                            c=ov_c, r=ov_r, n=ov_n, a=ov_a,
                            ttl_seconds=ttl,
                            source="space_ingestion",
                            reason=reason,
                        )
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def export_grid_summary(self, floorplan: dict) -> dict:
        """Return per-room CRNA averages and cell counts."""
        grid = get_grid()
        result: dict = {"rooms": {}}
        for room in floorplan.get("rooms", []):
            name = room.get("name", "unknown")
            bounds = room.get("bounds", {})
            x_range = bounds.get("x", [0, 1])
            y_range = bounds.get("y", [0, 1])
            z_range = bounds.get("z", [0, 1])
            cells = []
            for x in range(int(x_range[0]), int(x_range[1])):
                for y in range(int(y_range[0]), int(y_range[1])):
                    for z in range(int(z_range[0]), int(z_range[1])):
                        cells.append(grid.get(x, y, z))
            if cells:
                result["rooms"][name] = {
                    "cell_count": len(cells),
                    "avg_c": sum(v.c for v in cells) / len(cells),
                    "avg_r": sum(v.r for v in cells) / len(cells),
                    "avg_n": sum(v.n for v in cells) / len(cells),
                    "avg_a": sum(v.a for v in cells) / len(cells),
                }
        return result
