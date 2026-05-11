# MANIFOLD Physical v0.1 — QUICKSTART

Connect a real Roomba, a Home Assistant MQTT broker, and a webcam to MANIFOLD
in under 10 minutes.

---

## Prerequisites

```bash
# MANIFOLD server already running:
MANIFOLD_API_KEY=your-secret python -m manifold.server --port 8080

# Install optional physical layer dependencies:
pip install ultralytics          # YOLOv8 object detection (camera)
pip install opencv-python        # camera capture fallback
```

No external dependencies are required for the Roomba bridge or MQTT bridge —
they use only the Python standard library.

---

## 1. Connect a Roomba (iRobot REST cloud API)

```python
from manifold_physical.bridges.roomba_bridge import RoombaBridge

bridge = RoombaBridge(
    robot_id="roomba-01",
    cloud_email="you@example.com",
    cloud_password="your-irobot-cloud-password",
    grid_position=(0, 0, 0),           # where the Roomba starts in the CRNA grid
    manifold_url="http://localhost:8080",
    manifold_api_key="your-manifold-key",
)

if bridge.connect():
    bridge.start()   # starts command + sensor background threads
    print("Roomba is now governed by MANIFOLD")
```

The bridge:
- Registers the Roomba as a MANIFOLD agent at `POST /agents/register`
- Polls `GET /agents/{id}/commands` every 20 s and translates:
  - `pause` → stop cleaning
  - `resume` → start cleaning
  - `redirect` → set target region
- Polls iRobot state every 5 s; fires `SensorBridge.handle_obstacle` on bump
- Raises R in the CRNA grid cell at the robot's position

---

## 2. Connect a Home Assistant MQTT broker

```python
from manifold_physical.bridges.mqtt_bridge import MQTTBridge, DeviceMapping

# Option A — use the built-in Home Assistant profile
bridge = MQTTBridge(broker_host="homeassistant.local", broker_port=1883)
for dm in MQTTBridge.HomeAssistantProfile():
    bridge.subscribe(dm.topic, dm)
bridge.start()

# Option B — custom device mappings
bridge = MQTTBridge(broker_host="192.168.1.100")
bridge.subscribe("home/living_room/motion", DeviceMapping(
    topic="home/living_room/motion",
    device_type="motion_sensor",
    grid_coord=(3, 4, 0),
    risk_on_trigger=0.75,
))
bridge.start()
```

Minimum config JSON (for `POST /physical/init`):

```json
{
  "mqtt": {
    "broker_host": "homeassistant.local",
    "broker_port": 1883,
    "devices": [
      {
        "topic": "homeassistant/binary_sensor/+/motion/state",
        "device_type": "motion_sensor",
        "grid_coord": [2, 3, 0],
        "risk_on_trigger": 0.75
      }
    ]
  }
}
```

Supported device types: `motion_sensor`, `door_sensor`, `temperature`,
`smoke_detector`, `camera`.

---

## 3. Connect a webcam (YOLOv8 real-time obstacle detection)

```python
from manifold_physical.camera_detector import CameraDetector

# Built-in webcam with YOLOv8 nano
det = CameraDetector(
    camera_index=0,
    model_size="nano",
    grid_origin=(0, 0, 0),
    grid_scale=1.0,
    confidence_threshold=0.45,
)
det.start()

# On Raspberry Pi 5:
det = CameraDetector.RaspberryPiConfig()
det.start()
```

Detected objects are automatically mapped:
- `person` → obstacle_type `human`, R = 0.95
- `cat` / `dog` → obstacle_type `animal`, R = 0.85
- anything else → obstacle_type `physical_object`, R = 0.70

Minimum config JSON for `POST /physical/init`:

```json
{
  "cameras": [
    {
      "camera_index": 0,
      "model_size": "nano",
      "grid_origin": [0, 0, 0],
      "grid_scale": 1.0,
      "confidence_threshold": 0.45
    }
  ]
}
```

---

## 4. Initialise everything via the MANIFOLD API

You can also initialise the entire physical layer via a single API call:

```bash
curl -X POST http://localhost:8080/physical/init \
  -H "Content-Type: application/json" \
  -d @manifold_physical/config_example.json
```

Check status:

```bash
curl http://localhost:8080/physical/status
# {"roomba_connected": true, "mqtt_connected": true, "cameras_running": 1, ...}

curl http://localhost:8080/physical/cameras
# {"cameras": [{"id": "0", "status": "running", "detections_today": 12, ...}]}
```

---

## 5. Test without hardware

Every bridge supports `mock_mode=True` for hardware-free testing:

```python
from manifold_physical.physical_manager import PhysicalManager

pm = PhysicalManager(config={
    "roomba": {
        "robot_id": "roomba-test",
        "cloud_email": "test@example.com",
        "cloud_password": "secret",
        "mock_mode": True,
    }
})
pm.start_all()
print(pm.status())
pm.stop_all()
```

Run all tests:

```bash
pytest tests/ -q --tb=short
```

---

*MANIFOLD Physical v0.1 — Roomba · MQTT · Camera — all governed.*
