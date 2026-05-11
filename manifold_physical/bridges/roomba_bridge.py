"""RoombaBridge — connects an iRobot Roomba to MANIFOLD via the iRobot REST cloud API.

Zero external dependencies (uses stdlib urllib.request for HTTP).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from manifold_physical.sensor_bridge import ObstacleEvent, SensorBridge

_IROBOT_BASE_URL = "https://api.irobot.com"

# Number of sensor polls between simulated bump events in mock mode
MOCK_BUMP_INTERVAL = 7


class RoombaBridge:
    """Connects an iRobot Roomba to MANIFOLD as a governed physical agent.

    Parameters
    ----------
    robot_id:
        Unique identifier for this Roomba within MANIFOLD.
    cloud_email:
        iRobot cloud account email.
    cloud_password:
        iRobot cloud account password.
    grid_position:
        ``(x, y, z)`` tuple representing this robot's home position in the
        CRNA grid.
    manifold_url:
        Base URL of the MANIFOLD server to register with.
    manifold_api_key:
        MANIFOLD API key.
    mock_mode:
        When ``True``, all iRobot API calls are skipped and synthetic sensor
        data is used instead — suitable for unit tests without hardware.
    """

    CAPABILITIES = ["vacuum", "map", "floor_nav", "bump_detect"]
    DOMAIN = "physical/floor"

    def __init__(
        self,
        robot_id: str,
        cloud_email: str,
        cloud_password: str,
        grid_position: tuple = (0, 0, 0),
        manifold_url: str = "http://localhost:8080",
        manifold_api_key: str = "",
        mock_mode: bool = False,
    ) -> None:
        self.robot_id = robot_id
        self.cloud_email = cloud_email
        self.cloud_password = cloud_password
        self.grid_position = grid_position
        self.manifold_url = manifold_url.rstrip("/")
        self.manifold_api_key = manifold_api_key
        self.mock_mode = mock_mode

        self._sensor_bridge = SensorBridge()
        self._cloud_token: str | None = None
        self._running = False
        self._cmd_thread: threading.Thread | None = None
        self._sensor_thread: threading.Thread | None = None
        self._mock_poll_count = 0

    # ------------------------------------------------------------------
    # Connection / auth
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Authenticate with iRobot cloud and register with MANIFOLD.

        Returns ``True`` on success, ``False`` if authentication fails.
        """
        if self.mock_mode:
            self._cloud_token = "mock-token"
            self._register_with_manifold()
            return True
        try:
            self._cloud_token = self._irobot_auth()
            self._register_with_manifold()
            return True
        except Exception as exc:  # noqa: BLE001
            logging.error("RoombaBridge.connect failed: %s", exc)
            return False

    def _irobot_auth(self) -> str:
        """POST credentials to iRobot cloud, return bearer token."""
        payload = json.dumps({
            "email": self.cloud_email,
            "password": self.cloud_password,
        }).encode()
        req = urllib.request.Request(
            f"{_IROBOT_BASE_URL}/v1/auth",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["access_token"]

    def _register_with_manifold(self) -> None:
        """Register this Roomba as a MANIFOLD agent."""
        payload = json.dumps({
            "agent_id": self.robot_id,
            "name": f"Roomba {self.robot_id}",
            "capabilities": self.CAPABILITIES,
            "domain": self.DOMAIN,
            "type": "physical",
            "status": "active",
        }).encode()
        req = urllib.request.Request(
            f"{self.manifold_url}/agents/register",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.manifold_api_key}",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:  # noqa: BLE001
            logging.debug("RoombaBridge: MANIFOLD registration (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start command-polling and sensor-polling background threads."""
        self._running = True
        self._cmd_thread = threading.Thread(
            target=self._command_poll_loop,
            daemon=True,
            name=f"roomba-cmd-{self.robot_id}",
        )
        self._sensor_thread = threading.Thread(
            target=self._sensor_poll_loop,
            daemon=True,
            name=f"roomba-sensor-{self.robot_id}",
        )
        self._cmd_thread.start()
        self._sensor_thread.start()

    def stop(self) -> None:
        """Stop all background threads."""
        self._running = False

    # ------------------------------------------------------------------
    # Command polling (every 20 s)
    # ------------------------------------------------------------------

    def _command_poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_commands()
            except Exception as exc:  # noqa: BLE001
                logging.debug("RoombaBridge command poll error: %s", exc)
            time.sleep(20.0)

    def _poll_commands(self) -> None:
        """Fetch pending MANIFOLD commands and execute them on the Roomba."""
        if self.mock_mode:
            return
        req = urllib.request.Request(
            f"{self.manifold_url}/agents/{self.robot_id}/commands",
            headers={"Authorization": f"Bearer {self.manifold_api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            for cmd in data.get("commands", []):
                self._execute_command(cmd)
        except Exception as exc:  # noqa: BLE001
            logging.debug("RoombaBridge: command fetch failed: %s", exc)

    def _execute_command(self, cmd: dict) -> None:
        """Translate a MANIFOLD command to an iRobot API call."""
        action = cmd.get("action", "")
        try:
            if action == "pause":
                self._irobot_api("POST", "/v1/robot/stop_cleaning")
            elif action == "resume":
                self._irobot_api("POST", "/v1/robot/start_cleaning")
            elif action == "redirect":
                region = cmd.get("params", {}).get("region", {})
                self._irobot_api("POST", "/v1/robot/set_target", body=region)
            else:
                logging.debug("RoombaBridge: unknown command action '%s'", action)
        except Exception as exc:  # noqa: BLE001
            logging.error("RoombaBridge command '%s' execution error: %s", action, exc)

    # ------------------------------------------------------------------
    # Sensor polling (every 5 s)
    # ------------------------------------------------------------------

    def _sensor_poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_sensors()
            except Exception as exc:  # noqa: BLE001
                logging.debug("RoombaBridge sensor poll error: %s", exc)
            time.sleep(5.0)

    def _poll_sensors(self) -> None:
        """Read Roomba state and publish sensor events to MANIFOLD."""
        state = self._get_robot_state()
        if not state:
            return
        x = float(state.get("x", self.grid_position[0]))
        y = float(state.get("y", self.grid_position[1]))
        z = float(self.grid_position[2])

        # Bump sensors → obstacle event
        if state.get("bump_left") or state.get("bump_right"):
            self._sensor_bridge.handle_obstacle(ObstacleEvent(
                sensor_id=f"{self.robot_id}-bump",
                obstacle_type="physical_object",
                x=x,
                y=y,
                z=z,
                confidence=0.9,
            ))

        # Position update
        self._sensor_bridge.handle_robot_position(
            agent_id=self.robot_id,
            x=x,
            y=y,
            z=z,
        )

    def _get_robot_state(self) -> dict | None:
        """Return current Roomba state dict, or None on error."""
        if self.mock_mode:
            return self._mock_state()
        try:
            return self._irobot_api("GET", f"/v1/robot/{self.robot_id}/state")
        except Exception as exc:  # noqa: BLE001
            logging.error("RoombaBridge: state fetch failed: %s", exc)
            return None

    def _mock_state(self) -> dict:
        """Synthetic Roomba state for testing without real hardware."""
        self._mock_poll_count += 1
        return {
            "x": float(self.grid_position[0]) + (self._mock_poll_count * 0.1) % 5.0,
            "y": float(self.grid_position[1]),
            "bump_left": self._mock_poll_count % MOCK_BUMP_INTERVAL == 0,
            "bump_right": False,
            "battery": 95,
            "status": "cleaning",
        }

    # ------------------------------------------------------------------
    # iRobot REST API helper
    # ------------------------------------------------------------------

    def _irobot_api(self, method: str, path: str, body: dict | None = None) -> dict:
        """Make an authenticated call to the iRobot cloud REST API."""
        payload = json.dumps(body).encode() if body else None
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._cloud_token:
            headers["Authorization"] = f"Bearer {self._cloud_token}"
        if payload:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{_IROBOT_BASE_URL}{path}",
            data=payload,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
