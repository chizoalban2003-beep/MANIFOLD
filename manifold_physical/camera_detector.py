"""CameraDetector — real-time obstacle detection from a camera feed.

Uses YOLOv8 (ultralytics) when available; falls back to motion-detection-only
mode using opencv-python.  Both are optional — the module is always importable.

Detected obstacles are published to the CRNA bus via SensorBridge.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from manifold_physical.sensor_bridge import ObstacleEvent, SensorBridge

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------

try:
    from ultralytics import YOLO as _YOLO  # type: ignore[import-untyped]
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    _YOLO = None  # type: ignore[assignment,misc]

try:
    import cv2 as _cv2  # type: ignore[import-untyped]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    _cv2 = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# YOLO class → obstacle type + risk mapping
# ---------------------------------------------------------------------------

_CLASS_MAP: dict[str, tuple[str, float]] = {
    "person": ("human", 0.95),
    "cat": ("animal", 0.85),
    "dog": ("animal", 0.85),
}
_DEFAULT_OBSTACLE = ("physical_object", 0.70)

# Minimum seconds between obstacle events for the same grid cell
_MIN_EVENT_INTERVAL = 0.5
# Seconds of non-detection before handle_clear fires
_CLEAR_TIMEOUT = 3.0


@dataclass
class Detection:
    """One object detection result."""

    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coords
    estimated_grid_coord: tuple  # (gx, gy, gz) in CRNA grid


class CameraRegistry:
    """Singleton registry for all active CameraDetector instances."""

    _instance: "CameraRegistry | None" = None

    def __new__(cls) -> "CameraRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cameras: dict[str, "CameraDetector"] = {}
        return cls._instance

    def register(self, detector: "CameraDetector") -> None:
        self._cameras[str(detector.camera_index)] = detector

    def unregister(self, detector: "CameraDetector") -> None:
        self._cameras.pop(str(detector.camera_index), None)

    def status_list(self) -> list[dict]:
        result = []
        for cam_id, det in self._cameras.items():
            result.append({
                "id": cam_id,
                "status": "running" if det.is_running() else "stopped",
                "detections_today": det.detections_today,
                "last_detection": det.last_detection_time,
            })
        return result


def get_camera_registry() -> CameraRegistry:
    """Return the process-wide CameraRegistry singleton."""
    return CameraRegistry()


class CameraDetector:
    """Real-time camera-based obstacle detector.

    Parameters
    ----------
    camera_index:
        OpenCV camera index (0 = built-in webcam).
    model_size:
        YOLOv8 model variant: ``"nano"`` (fastest) or ``"small"``.
    grid_origin:
        ``(x, y, z)`` CRNA grid coordinate that maps to the top-left pixel.
    grid_scale:
        How many grid cells per metre in the camera's field of view.
    confidence_threshold:
        Minimum YOLO confidence score to process a detection.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_size: str = "nano",
        grid_origin: tuple = (0, 0, 0),
        grid_scale: float = 1.0,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.camera_index = camera_index
        self.model_size = model_size
        self.grid_origin = grid_origin
        self.grid_scale = grid_scale
        self.confidence_threshold = confidence_threshold

        self._sensor_bridge = SensorBridge()
        self._model: Any = None
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._running = False

        # Cell → last seen timestamp (for clear detection)
        self._last_seen: dict[tuple, float] = {}
        # Cell → last event timestamp (rate-limiting)
        self._last_event: dict[tuple, float] = {}

        self.detections_today: int = 0
        self.last_detection_time: float | None = None

        get_camera_registry().register(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def RaspberryPiConfig(cls) -> "CameraDetector":
        """Return a CameraDetector configured for Raspberry Pi 5."""
        return cls(
            camera_index=0,
            model_size="nano",
            grid_origin=(0, 0, 0),
            grid_scale=1.0,
            confidence_threshold=0.40,
        )

    def start(self) -> None:
        """Start the detection loop in a background thread."""
        self._running = True
        self._load_model()
        self._thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name=f"camera-detector-{self.camera_index}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the detection loop and release the camera."""
        self._running = False
        if self._cap is not None and _CV2_AVAILABLE:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
        get_camera_registry().unregister(self)

    def is_running(self) -> bool:
        """Return True if the detection loop is active."""
        return self._running

    def detect_objects(self, frame: Any) -> list[Detection]:
        """Process one frame and return a list of Detection objects.

        Parameters
        ----------
        frame:
            A numpy array (BGR image from OpenCV) or None for a blank frame.
        """
        if self._model is not None:
            return self._detect_with_yolo(frame)
        if _CV2_AVAILABLE and frame is not None:
            return self._detect_with_motion(frame)
        return []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load YOLOv8 model, or fall back to motion detection."""
        if _YOLO_AVAILABLE:
            model_name = f"yolov8{self.model_size[0]}.pt"
            try:
                self._model = _YOLO(model_name)
                logging.debug("CameraDetector: loaded %s", model_name)
                return
            except Exception as exc:  # noqa: BLE001
                logging.warning("CameraDetector: YOLO load failed (%s), using motion detection", exc)
        self._model = None
        if not _CV2_AVAILABLE:
            logging.warning(
                "CameraDetector: neither ultralytics nor opencv-python installed. "
                "No detections will be produced."
            )

    # ------------------------------------------------------------------
    # Detection implementations
    # ------------------------------------------------------------------

    def _detect_with_yolo(self, frame: Any) -> list[Detection]:
        """Run YOLOv8 inference on *frame*."""
        try:
            results = self._model(frame, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logging.debug("CameraDetector YOLO inference error: %s", exc)
            return []
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                cls_name = result.names.get(cls_id, "object")
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                grid_coord = self._pixel_to_grid(cx, cy)
                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    estimated_grid_coord=grid_coord,
                ))
        return detections

    def _detect_with_motion(self, frame: Any) -> list[Detection]:
        """Basic motion-detection fallback using opencv frame differencing."""
        # Simplified: convert to grey, blur, check mean intensity change
        try:
            grey = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            grey = _cv2.GaussianBlur(grey, (21, 21), 0)
            if not hasattr(self, "_prev_frame"):
                self._prev_frame = grey
                return []
            delta = _cv2.absdiff(self._prev_frame, grey)
            self._prev_frame = grey
            thresh = _cv2.threshold(delta, 25, 255, _cv2.THRESH_BINARY)[1]
            if thresh.mean() > 1.0:
                h, w = frame.shape[:2]
                grid_coord = self._pixel_to_grid(w / 2.0, h / 2.0)
                return [Detection(
                    class_name="motion",
                    confidence=0.5,
                    bbox=(0.0, 0.0, float(w), float(h)),
                    estimated_grid_coord=grid_coord,
                )]
        except Exception as exc:  # noqa: BLE001
            logging.debug("CameraDetector motion detection error: %s", exc)
        return []

    def _pixel_to_grid(self, px: float, py: float) -> tuple:
        """Convert pixel centre coordinates to CRNA grid cell."""
        gx = int(self.grid_origin[0] + px / (100.0 * self.grid_scale))
        gy = int(self.grid_origin[1] + py / (100.0 * self.grid_scale))
        gz = int(self.grid_origin[2])
        return (max(0, gx), max(0, gy), max(0, gz))

    # ------------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Continuously read frames from camera and publish obstacle events."""
        if _CV2_AVAILABLE:
            self._cap = _cv2.VideoCapture(self.camera_index)

        while self._running:
            frame = self._read_frame()
            detections = self.detect_objects(frame)
            now = time.monotonic()

            seen_cells: set[tuple] = set()
            for det in detections:
                cell = det.estimated_grid_coord
                seen_cells.add(cell)
                # Rate-limit: skip if we fired an event for this cell too recently
                if now - self._last_event.get(cell, 0.0) < _MIN_EVENT_INTERVAL:
                    self._last_seen[cell] = now
                    continue
                self._last_event[cell] = now
                self._last_seen[cell] = now
                self.detections_today += 1
                self.last_detection_time = now
                obstacle_type, risk = _CLASS_MAP.get(det.class_name, _DEFAULT_OBSTACLE)
                self._sensor_bridge.handle_obstacle(ObstacleEvent(
                    sensor_id=f"camera-{self.camera_index}",
                    obstacle_type=obstacle_type,
                    x=float(cell[0]),
                    y=float(cell[1]),
                    z=float(cell[2]),
                    confidence=risk * det.confidence,
                    ttl=_CLEAR_TIMEOUT,
                ))

            # Fire handle_clear for cells not seen recently
            stale = [
                cell for cell, ts in self._last_seen.items()
                if cell not in seen_cells and now - ts >= _CLEAR_TIMEOUT
            ]
            for cell in stale:
                self._sensor_bridge.handle_clear(
                    sensor_id=f"camera-{self.camera_index}",
                    x=float(cell[0]),
                    y=float(cell[1]),
                    z=float(cell[2]),
                )
                del self._last_seen[cell]
                self._last_event.pop(cell, None)

            time.sleep(0.1)

    def _read_frame(self) -> Any:
        """Read one frame from the camera, returning None on failure."""
        if self._cap is None:
            return None
        try:
            ret, frame = self._cap.read()
            return frame if ret else None
        except Exception:  # noqa: BLE001
            return None
