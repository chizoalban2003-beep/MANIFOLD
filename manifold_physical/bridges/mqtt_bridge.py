"""MQTTBridge — connects any MQTT IoT device to MANIFOLD.

Implements a minimal MQTT 3.1.1 client using only the Python standard
library (``socket`` + ``struct``).  No external packages required.

Works with Home Assistant, Philips Hue, Zigbee, Z-Wave, and any device
that speaks MQTT 3.1.1 over TCP.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from manifold_physical.sensor_bridge import ObstacleEvent, SensorBridge

# Payloads that indicate an idle / clear state
_IDLE_PAYLOADS = frozenset({"off", "0", "closed", "idle", "clear", "false", "none"})

# device_type → (obstacle_type, risk_override | None)
# risk_override overrides mapping.risk_on_trigger when not None
_DEVICE_OBSTACLE_MAP: dict[str, tuple[str, float | None]] = {
    "motion_sensor": ("human", None),
    "camera": ("human", None),
    "smoke_detector": ("physical_object", 0.95),
    "door_sensor": ("unknown", None),
    "temperature": ("unknown", None),
}

_AGENT_TOPIC_PREFIX = "manifold/agent"
_MIN_HEARTBEAT_INTERVAL = 0.05
_RECONNECT_BACKOFF_INITIAL = 0.25
_RECONNECT_BACKOFF_MAX = 5.0


@dataclass
class DeviceMapping:
    """Maps one MQTT topic to a CRNA grid position and risk level.

    Parameters
    ----------
    topic:
        MQTT topic string.  Supports ``+`` (single-level wildcard) and
        ``#`` (multi-level wildcard) per the MQTT 3.1.1 specification.
    device_type:
        One of ``"motion_sensor"``, ``"door_sensor"``, ``"temperature"``,
        ``"smoke_detector"``, or ``"camera"``.
    grid_coord:
        ``(x, y, z)`` tuple in the CRNA grid for this device.
    risk_on_trigger:
        Risk delta to apply to the CRNA cell when the device triggers.
    """

    topic: str
    device_type: Literal[
        "motion_sensor", "door_sensor", "temperature", "smoke_detector", "camera"
    ]
    grid_coord: tuple = (0, 0, 0)
    risk_on_trigger: float = 0.75


class MQTTBridge:
    """Connects MQTT IoT devices to MANIFOLD's CRNA bus.

    Parameters
    ----------
    broker_host:
        Hostname or IP of the MQTT broker.
    broker_port:
        TCP port of the MQTT broker (default: 1883).
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        *,
        agent_id: str | None = None,
        command_callback: Callable[[dict[str, Any]], None] | None = None,
        heartbeat_interval: float = 0.5,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.agent_id = self._validate_agent_id(agent_id)
        self.command_callback = command_callback
        self.heartbeat_interval = max(_MIN_HEARTBEAT_INTERVAL, float(heartbeat_interval))
        self._mappings: dict[str, DeviceMapping] = {}
        self._sensor_bridge = SensorBridge()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._heartbeat_running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._packet_id = 0
        self._heartbeat_counter = 0
        self._heartbeat_lock = threading.Lock()
        self._socket_lock = threading.RLock()
        self._publish_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, mapping: DeviceMapping) -> None:
        """Register a topic → DeviceMapping binding.

        If called before ``start()``, the subscription is sent to the broker
        once the connection is established.  May also be called after
        ``start()`` to subscribe to additional topics.
        """
        self._mappings[topic] = mapping
        if self._sock is not None and self._running:
            try:
                self._send_subscribe(self._sock, topic)
            except Exception as exc:  # noqa: BLE001
                logging.debug("MQTTBridge: live subscribe failed: %s", exc)

    def set_agent(self, agent_id: str) -> None:
        """Bind the bridge to an agent-id topic namespace."""
        self.agent_id = self._validate_agent_id(agent_id)
        if self._sock is not None and self._running:
            self._subscribe_runtime_topics(self._sock)

    @staticmethod
    def _validate_agent_id(agent_id: str | None) -> str | None:
        """Reject agent IDs that would create invalid MQTT topic paths."""
        if agent_id is None:
            return None
        if any(ch in agent_id for ch in "/+#\x00"):
            raise ValueError("agent_id may not contain '/', '+', '#', or NUL")
        return agent_id

    def start(self) -> None:
        """Connect to the broker and start the receive loop in a background thread."""
        self._running = True
        self._sock = self._connect_to_broker()
        if self._sock is None:
            self._running = False
            return
        self._thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name=f"mqtt-recv-{self.broker_host}",
        )
        self._thread.start()
        if self.agent_id is not None:
            self.start_heartbeat()

    def disconnect(self) -> None:
        """Stop the receive loop and close the socket."""
        self._running = False
        self.stop_heartbeat()
        self._close_socket()

    def is_connected(self) -> bool:
        """Return True if the broker connection is active."""
        return self._running and self._sock is not None

    @classmethod
    def HomeAssistantProfile(cls) -> list[DeviceMapping]:
        """Return DeviceMapping objects for common Home Assistant MQTT topics."""
        return [
            DeviceMapping(
                topic="homeassistant/binary_sensor/+/motion/state",
                device_type="motion_sensor",
                grid_coord=(0, 0, 0),
                risk_on_trigger=0.75,
            ),
            DeviceMapping(
                topic="homeassistant/binary_sensor/+/door/state",
                device_type="door_sensor",
                grid_coord=(0, 0, 0),
                risk_on_trigger=0.60,
            ),
            DeviceMapping(
                topic="homeassistant/sensor/+/smoke/state",
                device_type="smoke_detector",
                grid_coord=(0, 0, 0),
                risk_on_trigger=0.95,
            ),
        ]

    @property
    def agent_cmd_topic(self) -> str | None:
        """MQTT topic for inbound commands for the bound agent."""
        if self.agent_id is None:
            return None
        return f"{_AGENT_TOPIC_PREFIX}/{self.agent_id}/cmd"

    @property
    def agent_telemetry_topic(self) -> str | None:
        """MQTT topic for outbound telemetry for the bound agent."""
        if self.agent_id is None:
            return None
        return f"{_AGENT_TOPIC_PREFIX}/{self.agent_id}/telemetry"

    @property
    def agent_status_topic(self) -> str | None:
        """MQTT topic for outbound system health/status for the bound agent."""
        if self.agent_id is None:
            return None
        return f"{_AGENT_TOPIC_PREFIX}/{self.agent_id}/status"

    def publish_telemetry(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        *,
        state: str,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Publish the robot's live position/status on the telemetry topic."""
        payload: dict[str, Any] = {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "z": round(float(z), 3),
            "state": state,
        }
        if extra:
            payload.update(extra)
        return self._publish_json(self.agent_telemetry_topic, payload)

    def publish_status(
        self,
        *,
        state: str,
        battery: float | int | None = None,
        heartbeat: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Publish robot health to the status topic."""
        if heartbeat is None:
            with self._heartbeat_lock:
                heartbeat = self._heartbeat_counter
        payload: dict[str, Any] = {
            "state": state,
            "heartbeat": heartbeat,
            "connected": self.is_connected(),
        }
        if battery is not None:
            payload["battery"] = battery
        if extra:
            payload.update(extra)
        return self._publish_json(self.agent_status_topic, payload)

    def publish_command(
        self,
        command: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a command to the agent command topic."""
        body: dict[str, Any] = {"command": command}
        if payload:
            body.update(payload)
        return self._publish_json(self.agent_cmd_topic, body)

    def start_heartbeat(self, interval: float | None = None) -> None:
        """Start the periodic status heartbeat publisher."""
        if interval is not None:
            self.heartbeat_interval = max(_MIN_HEARTBEAT_INTERVAL, float(interval))
        if self._heartbeat_running or self.agent_id is None:
            return
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"mqtt-heartbeat-{self.agent_id}",
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop the periodic status heartbeat publisher."""
        self._heartbeat_running = False

    def heartbeat_tick(self, *, state: str = "active", extra: dict[str, Any] | None = None) -> bool:
        """Publish one heartbeat sample and increment the counter."""
        with self._heartbeat_lock:
            self._heartbeat_counter += 1
            heartbeat = self._heartbeat_counter
        return self.publish_status(state=state, heartbeat=heartbeat, extra=extra)

    # ------------------------------------------------------------------
    # MQTT 3.1.1 minimal client — connection
    # ------------------------------------------------------------------

    def _connect_to_broker(self) -> socket.socket | None:
        """Open TCP connection, send CONNECT, receive CONNACK, subscribe."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self.broker_host, self.broker_port))
            self._send_connect(sock)
            if not self._recv_connack(sock):
                sock.close()
                return None
            sock.settimeout(None)  # blocking receive in background thread
            self._subscribe_runtime_topics(sock)
            return sock
        except Exception as exc:  # noqa: BLE001
            logging.error("MQTTBridge: connect to %s:%s failed: %s",
                          self.broker_host, self.broker_port, exc)
            return None

    def _subscribe_runtime_topics(self, sock: socket.socket) -> None:
        """Subscribe to configured sensor topics and the agent command topic."""
        for topic in self._mappings:
            self._send_subscribe(sock, topic)
        if self.agent_cmd_topic is not None:
            self._send_subscribe(sock, self.agent_cmd_topic)

    def _close_socket(self) -> None:
        """Close the current socket, swallowing transport errors."""
        with self._socket_lock:
            sock = self._sock
            self._sock = None
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            pass

    def _publish_json(self, topic: str | None, payload: dict[str, Any]) -> bool:
        """Serialize *payload* to JSON and publish to *topic*."""
        try:
            message = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            with self._socket_lock:
                sock = self._sock
            if topic is None or sock is None or not self._running:
                return False
            self._publish(sock, topic, message)
            return True
        except Exception as exc:  # noqa: BLE001
            logging.debug("MQTTBridge publish failed: %s", exc)
            return False

    def _publish(self, sock: socket.socket, topic: str, payload: str) -> None:
        """Publish a UTF-8 payload to *topic* over the current socket."""
        topic_bytes = topic.encode()
        payload_bytes = payload.encode()
        variable_header = struct.pack("!H", len(topic_bytes)) + topic_bytes
        body = variable_header + payload_bytes
        with self._publish_lock:
            sock.sendall(b"\x30" + _encode_remaining(len(body)) + body)

    def _send_connect(self, sock: socket.socket) -> None:
        """Assemble and send an MQTT 3.1.1 CONNECT packet."""
        client_id = b"manifold-mqtt"
        # Protocol name (UTF-8 "MQTT") + level (4) + flags (clean session) + keep-alive (60 s)
        variable_header = (
            b"\x00\x04MQTT"   # protocol name
            b"\x04"           # protocol level = 3.1.1
            b"\x02"           # connect flags: clean session only
            + struct.pack("!H", 60)  # keep-alive seconds
        )
        payload = struct.pack("!H", len(client_id)) + client_id
        body = variable_header + payload
        sock.sendall(b"\x10" + _encode_remaining(len(body)) + body)

    def _recv_connack(self, sock: socket.socket) -> bool:
        """Read CONNACK and return True if the connection was accepted."""
        try:
            header = _recv_exact(sock, 2)
            if not header or header[0] != 0x20:
                logging.error("MQTTBridge: expected CONNACK (0x20), got 0x%02x", header[0] if header else 0)
                return False
            remaining = header[1]
            payload = _recv_exact(sock, remaining)
            if payload and len(payload) >= 2 and payload[1] != 0:
                logging.error("MQTTBridge: CONNACK return code %d (rejected)", payload[1])
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            logging.error("MQTTBridge: CONNACK read error: %s", exc)
            return False

    def _next_packet_id(self) -> int:
        self._packet_id = (self._packet_id % 65535) + 1
        return self._packet_id

    def _send_subscribe(self, sock: socket.socket, topic: str) -> None:
        """Send an MQTT SUBSCRIBE packet for the given topic (QoS 0)."""
        topic_bytes = topic.encode()
        pid = self._next_packet_id()
        payload = struct.pack("!H", len(topic_bytes)) + topic_bytes + b"\x00"  # QoS 0
        variable_header = struct.pack("!H", pid)
        body = variable_header + payload
        sock.sendall(b"\x82" + _encode_remaining(len(body)) + body)

    # ------------------------------------------------------------------
    # MQTT 3.1.1 minimal client — receive loop
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        backoff = _RECONNECT_BACKOFF_INITIAL
        while self._running:
            with self._socket_lock:
                sock = self._sock
            if sock is None:
                sock = self._connect_to_broker()
                with self._socket_lock:
                    self._sock = sock
                if sock is None:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, _RECONNECT_BACKOFF_MAX)
                    continue
                backoff = _RECONNECT_BACKOFF_INITIAL
            try:
                packet = self._recv_packet()
                if packet is None:
                    raise OSError("MQTT connection dropped")
                self._handle_packet(packet)
            except OSError:
                logging.debug("MQTTBridge: connection dropped, reconnecting")
                self._close_socket()
                if not self._running:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2.0, _RECONNECT_BACKOFF_MAX)
            except Exception as exc:  # noqa: BLE001
                logging.debug("MQTTBridge recv error: %s", exc)

    def _recv_packet(self) -> tuple[int, bytes] | None:
        """Read one complete MQTT packet from the socket."""
        with self._socket_lock:
            sock = self._sock
        if not sock:
            return None
        first = _recv_exact(sock, 1)
        if not first:
            return None
        packet_type = first[0]
        remaining = _decode_remaining(sock)
        body = _recv_exact(sock, remaining) if remaining > 0 else b""
        if body is None:
            return None
        return (packet_type, body)

    def _handle_packet(self, packet: tuple[int, bytes]) -> None:
        ptype, body = packet
        if ptype & 0xF0 == 0x30:  # PUBLISH
            self._handle_publish(ptype, body)
        elif ptype == 0x90:   # SUBACK — ignore
            pass
        elif ptype == 0xD0:   # PINGRESP — ignore
            pass
        # All other packet types silently ignored

    def _handle_publish(self, ptype: int, body: bytes) -> None:
        """Parse an MQTT PUBLISH packet and fire the appropriate sensor event."""
        if len(body) < 2:
            return
        offset = 0
        topic_len = struct.unpack_from("!H", body, offset)[0]
        offset += 2
        if offset + topic_len > len(body):
            return
        topic = body[offset:offset + topic_len].decode(errors="replace")
        offset += topic_len
        qos = (ptype >> 1) & 0x03
        if qos > 0:
            offset += 2  # skip packet identifier
        payload_text = body[offset:].decode(errors="replace").strip()
        payload_raw = payload_text.lower()

        if topic == self.agent_cmd_topic:
            self._handle_agent_command(payload_text)
            return

        mapping = self._match_topic(topic)
        if mapping is None:
            return

        if payload_raw in _IDLE_PAYLOADS:
            self._sensor_bridge.handle_clear(
                sensor_id=f"mqtt-{mapping.topic}",
                x=float(mapping.grid_coord[0]),
                y=float(mapping.grid_coord[1]),
                z=float(mapping.grid_coord[2]),
            )
            return

        obstacle_type, risk_override = _DEVICE_OBSTACLE_MAP.get(
            mapping.device_type, ("unknown", None)
        )
        risk = risk_override if risk_override is not None else mapping.risk_on_trigger
        self._sensor_bridge.handle_obstacle(ObstacleEvent(
            sensor_id=f"mqtt-{mapping.topic}",
            obstacle_type=obstacle_type,
            x=float(mapping.grid_coord[0]),
            y=float(mapping.grid_coord[1]),
            z=float(mapping.grid_coord[2]),
            confidence=risk,
            ttl=30.0,
        ))

    def _handle_agent_command(self, payload_text: str) -> None:
        """Parse a command payload and hand it to the configured callback."""
        try:
            command = json.loads(payload_text)
        except json.JSONDecodeError:
            command = {"command": payload_text}
        if isinstance(command, str):
            command = {"command": command}
        if not isinstance(command, dict):
            command = {"command": "UNKNOWN", "payload": command}
        normalized = dict(command)
        normalized.setdefault("agent_id", self.agent_id)
        normalized.setdefault("topic", self.agent_cmd_topic)
        if self.command_callback is not None:
            try:
                self.command_callback(normalized)
            except Exception as exc:  # noqa: BLE001
                logging.debug("MQTTBridge command callback failed: %s", exc)

    def _heartbeat_loop(self) -> None:
        """Background loop publishing a status heartbeat every interval."""
        while self._running and self._heartbeat_running:
            try:
                self.heartbeat_tick()
            except Exception as exc:  # noqa: BLE001
                logging.debug("MQTTBridge heartbeat failed: %s", exc)
            time.sleep(self.heartbeat_interval)

    def _match_topic(self, topic: str) -> DeviceMapping | None:
        """Match an incoming topic against registered patterns (supports + and #)."""
        if topic in self._mappings:
            return self._mappings[topic]
        for pattern, mapping in self._mappings.items():
            if _topic_matches(pattern, topic):
                return mapping
        return None

    # ------------------------------------------------------------------
    # Expose internal helpers for testing
    # ------------------------------------------------------------------

    def _simulate_publish(self, topic: str, payload: str) -> None:
        """Simulate receiving a PUBLISH on *topic* with *payload*.

        Used in unit tests to avoid a real broker connection.
        """
        topic_bytes = topic.encode()
        payload_bytes = payload.encode()
        # ptype = 0x30 (PUBLISH, QoS 0, no DUP, no RETAIN)
        body = struct.pack("!H", len(topic_bytes)) + topic_bytes + payload_bytes
        self._handle_publish(0x30, body)

    def _simulate_command(self, payload: str) -> None:
        """Convenience helper for tests: simulate an inbound command payload."""
        if self.agent_cmd_topic is None:
            raise ValueError("agent_id must be set before simulating commands")
        self._simulate_publish(self.agent_cmd_topic, payload)


@dataclass(slots=True)
class ManifoldMQTTGateway:
    """Output gateway that syncs a brain's live state to MQTT topics."""

    brain: Any
    bridge: MQTTBridge
    agent_id: str | None = None

    def __post_init__(self) -> None:
        """Wire the bridge's command_callback to route policy commands to the brain."""
        if self.bridge.command_callback is None:
            self.bridge.command_callback = self._on_command

    def _on_command(self, command: dict[str, Any]) -> None:
        """Route inbound MQTT commands to the brain's policy handler."""
        handle_command = getattr(self.brain, "handle_command", None)
        if handle_command is None:
            return
        action_code = command.get("action_code")
        if action_code is not None:
            agent_id = command.get("agent_id")
            handle_command(int(action_code), command.get("params") or {}, agent_id=agent_id)

    def sync_to_hardware(self) -> bool:
        """Push the brain's current physical state to MQTT telemetry/status."""
        agent_id = self.agent_id or self.bridge.agent_id
        if agent_id is None:
            return False
        if self.bridge.agent_id != agent_id:
            self.bridge.set_agent(agent_id)

        physical_pos = getattr(self.brain, "physical_pos", (0.0, 0.0, 0.0))
        movement_state = getattr(self.brain, "movement_state", None)
        state_name = getattr(movement_state, "name", None) or str(movement_state or "IDLE")
        logical_pos = getattr(self.brain, "logical_pos", physical_pos)

        telemetry_ok = self.bridge.publish_telemetry(
            physical_pos[0],
            physical_pos[1],
            physical_pos[2] if len(physical_pos) > 2 else 0.0,
            state=state_name,
            extra={
                "logical_pos": list(logical_pos),
                "replan_pending": bool(getattr(self.brain, "replan_pending", False)),
            },
        )
        status_ok = self.bridge.publish_status(
            state=state_name,
            extra={
                "logical_pos": list(logical_pos),
                "physical_pos": list(physical_pos),
            },
        )
        return telemetry_ok and status_ok


# ------------------------------------------------------------------
# MQTT variable-length encoding helpers
# ------------------------------------------------------------------

def _encode_remaining(length: int) -> bytes:
    """Encode an integer as MQTT variable-length remaining-length bytes."""
    if length == 0:
        return b"\x00"
    encoded = []
    while length > 0:
        byte = length % 128
        length //= 128
        if length > 0:
            byte |= 0x80
        encoded.append(byte)
    return bytes(encoded)


def _decode_remaining(sock: socket.socket) -> int:
    """Read MQTT variable-length remaining-length from socket."""
    multiplier = 1
    value = 0
    for _ in range(4):
        data = _recv_exact(sock, 1)
        if not data:
            return value
        b = data[0]
        value += (b & 0x7F) * multiplier
        if (b & 0x80) == 0:
            break
        multiplier *= 128
    return value


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly *n* bytes from *sock*, returning None on EOF."""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def _topic_matches(pattern: str, topic: str) -> bool:
    """Return True if *topic* matches MQTT wildcard *pattern*."""
    pattern_parts = pattern.split("/")
    topic_parts = topic.split("/")
    if pattern_parts[-1] == "#":
        prefix = pattern_parts[:-1]
        return topic_parts[:len(prefix)] == prefix
    if len(pattern_parts) != len(topic_parts):
        return False
    return all(p == t or p == "+" for p, t in zip(pattern_parts, topic_parts))
