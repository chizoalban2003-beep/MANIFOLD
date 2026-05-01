"""Live event connectors for MANIFOLD layered state updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConnectorEvent:
    """External event projected into the layered grid."""

    generation: int
    layer: str
    cell: int
    delta: float
    note: str = ""


def load_connector_events(path: Path | None) -> tuple[ConnectorEvent, ...]:
    """Load CSV events from disk.

    CSV format (header required):
        generation,layer,cell,delta,note

    Layer values:
        physical_risk | info_noise | social_reputation
    """

    if path is None:
        return ()
    if not path.exists():
        raise FileNotFoundError(f"Connector events file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ()
    header = [part.strip() for part in lines[0].split(",")]
    if header[:4] != ["generation", "layer", "cell", "delta"]:
        raise ValueError(
            "Connector CSV header must start with: generation,layer,cell,delta"
        )

    events: list[ConnectorEvent] = []
    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 4:
            raise ValueError(f"Malformed connector event line: {raw!r}")
        note = parts[4] if len(parts) > 4 else ""
        events.append(
            ConnectorEvent(
                generation=int(parts[0]),
                layer=parts[1],
                cell=int(parts[2]),
                delta=float(parts[3]),
                note=note,
            )
        )
    return tuple(events)
