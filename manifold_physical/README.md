# MANIFOLD Physical

MANIFOLD Physical converts real physical spaces into governed CRNA grids.

Robots operating in these spaces are governed by MANIFOLD — every
movement step is risk-priced before execution.

## Overview

Physical spaces (homes, factories, hospitals, offices) are described
as JSON floor plans and ingested via `SpaceIngestion`.  Each room,
obstacle, and path becomes a set of CRNA-valued cells in `DynamicGrid`.

Robots receive governed CRNA values before every movement step.  When
sensors detect obstacles, `SensorBridge` translates those events into
`CellUpdate` messages on `CellUpdateBus`, which immediately raises Risk
in the affected cells so the planner can route around them.

## Quick start

```python
from manifold_physical.space_ingestion import SpaceIngestion

ingestion = SpaceIngestion()
floorplan = ingestion.load_floorplan("my_home.json")
cells = ingestion.ingest(floorplan)
print(f"Populated {cells} CRNA cells")
```

```python
from manifold_physical.sensor_bridge import SensorBridge, ObstacleEvent

bridge = SensorBridge()
bridge.handle_obstacle(ObstacleEvent(
    sensor_id="lidar-01",
    obstacle_type="human",
    x=3.0, y=2.0,
    confidence=0.95,
))
```
