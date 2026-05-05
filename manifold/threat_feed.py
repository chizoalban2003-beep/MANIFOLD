"""Phase 34: Threat Intelligence Feed — Real-Time HTTP Event Stream.

The ``ThreatFeedStreamer`` converts ``FederatedGossipPacket`` records and
canary probe failures into standardised Threat-Intel JSON payloads that can
be consumed by enterprise firewalls or SIEM systems via a long-lived
Server-Sent Events (SSE) endpoint.

The companion ``GET /feed`` route in ``server.py`` streams these events using
``Transfer-Encoding: chunked`` and ``Content-Type: text/event-stream``.

Key classes
-----------
``ThreatIntelPayload``
    Immutable, JSON-serialisable threat intelligence event.
``ThreatFeedStreamer``
    Ingests gossip packets and canary results; maintains a ring-buffer of
    recent events; exposes an SSE-formatted generator.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generator

from .federation import FederatedGossipPacket
from .probe import CanaryResult


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_GOSSIP_SEVERITY: dict[str, str] = {
    "failing": "high",
    "degraded": "medium",
    "healthy": "low",
}


def _canary_severity(result: CanaryResult) -> str:
    """Map a :class:`~manifold.probe.CanaryResult` probe action to a severity."""
    if result.probe_action == "suspect":
        return "critical"
    if result.probe_action == "fail":
        return "high"
    return "low"


# ---------------------------------------------------------------------------
# ThreatIntelPayload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThreatIntelPayload:
    """Standardised threat intelligence event record.

    Attributes
    ----------
    event_type:
        Category of the event.  One of:
        ``"gossip_failing"``, ``"gossip_degraded"``, ``"canary_fail"``,
        ``"canary_suspect"``, ``"canary_pass"``.
    tool_name:
        The tool or org that triggered the event.
    severity:
        ``"low"`` | ``"medium"`` | ``"high"`` | ``"critical"``.
    timestamp:
        POSIX timestamp of the event.
    details:
        Additional event-specific context.
    source:
        Originating system (e.g. ``"gossip_bridge"`` or ``"canary_prober"``).
    """

    event_type: str
    tool_name: str
    severity: str
    timestamp: float
    details: dict[str, Any]
    source: str = "manifold"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "details": self.details,
            "source": self.source,
        }

    def to_json(self) -> str:
        """Return a compact JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_sse(self) -> str:
        """Return an SSE-formatted string (``data: ...\\n\\n``)."""
        return f"data: {self.to_json()}\n\n"


# ---------------------------------------------------------------------------
# ThreatFeedStreamer
# ---------------------------------------------------------------------------


@dataclass
class ThreatFeedStreamer:
    """Ingests gossip packets and canary results and maintains a ring-buffer
    of :class:`ThreatIntelPayload` events for SSE streaming.

    Only **actionable** events are stored (healthy gossip and passing canaries
    are suppressed by default via ``include_low_severity``).

    Parameters
    ----------
    max_events:
        Maximum number of events to retain in the ring-buffer.  Older events
        are discarded when the buffer is full.  Default: ``500``.
    include_low_severity:
        When ``True``, low-severity events (healthy gossip, passing canaries)
        are also stored.  Default: ``False``.
    clock:
        Callable returning the current POSIX timestamp.  Override in tests.

    Example
    -------
    ::

        streamer = ThreatFeedStreamer()
        streamer.ingest_gossip(packet)    # from FederatedGossipBridge
        streamer.ingest_canary(result)    # from ActiveProber
        for event in streamer.recent_events(n=10):
            print(event.to_sse())
    """

    max_events: int = 500
    include_low_severity: bool = False
    clock: Any = None  # callable | None

    _events: list[ThreatIntelPayload] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = time.time

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_gossip(
        self,
        packet: FederatedGossipPacket,
    ) -> ThreatIntelPayload | None:
        """Convert a :class:`~manifold.federation.FederatedGossipPacket` into
        a :class:`ThreatIntelPayload` and store it.

        Only ``"failing"`` and ``"degraded"`` signals are stored unless
        ``include_low_severity`` is ``True``.

        Parameters
        ----------
        packet:
            The gossip packet to ingest.

        Returns
        -------
        ThreatIntelPayload | None
            The created payload, or ``None`` if the event was suppressed.
        """
        severity = _GOSSIP_SEVERITY.get(packet.signal, "medium")
        if severity == "low" and not self.include_low_severity:
            return None

        event_type = f"gossip_{packet.signal}"
        payload = ThreatIntelPayload(
            event_type=event_type,
            tool_name=packet.tool_name,
            severity=severity,
            timestamp=self.clock(),  # type: ignore[operator]
            details={
                "signal": packet.signal,
                "confidence": packet.confidence,
                "org_id": packet.org_id,
                "weight": packet.weight,
            },
            source="gossip_bridge",
        )
        self._store(payload)
        return payload

    def ingest_canary(
        self,
        result: CanaryResult,
    ) -> ThreatIntelPayload | None:
        """Convert a :class:`~manifold.probe.CanaryResult` into a
        :class:`ThreatIntelPayload` and store it.

        Passing canaries (``probe_action == "pass"``) are suppressed unless
        ``include_low_severity`` is ``True``.

        Parameters
        ----------
        result:
            The canary probe result to ingest.

        Returns
        -------
        ThreatIntelPayload | None
            The created payload, or ``None`` if the event was suppressed.
        """
        severity = _canary_severity(result)
        if severity == "low" and not self.include_low_severity:
            return None

        event_type = f"canary_{result.probe_action}"
        payload = ThreatIntelPayload(
            event_type=event_type,
            tool_name=result.tool_name,
            severity=severity,
            timestamp=result.timestamp,
            details={
                "probe_action": result.probe_action,
                "entropy_score_before": result.entropy_score_before,
                "adversarial_suspect": result.adversarial_suspect,
                "penalty_applied": result.penalty_applied,
            },
            source="canary_prober",
        )
        self._store(payload)
        return payload

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recent_events(self, n: int = 50) -> list[ThreatIntelPayload]:
        """Return the *n* most recent events (newest first).

        Parameters
        ----------
        n:
            Maximum number of events to return.

        Returns
        -------
        list[ThreatIntelPayload]
        """
        with self._lock:
            return list(reversed(self._events[-n:])) if self._events else []

    def events_by_severity(self, severity: str) -> list[ThreatIntelPayload]:
        """Return all stored events with the given *severity*.

        Parameters
        ----------
        severity:
            ``"low"``, ``"medium"``, ``"high"``, or ``"critical"``.

        Returns
        -------
        list[ThreatIntelPayload]
        """
        with self._lock:
            return [e for e in self._events if e.severity == severity]

    def event_count(self) -> int:
        """Return total number of stored events."""
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """Clear all stored events."""
        with self._lock:
            self._events.clear()

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    def sse_stream(
        self,
        poll_interval: float = 1.0,
        max_events: int | None = None,
    ) -> Generator[str, None, None]:
        """Generator that yields SSE-formatted strings for all stored events.

        This is a **snapshot** generator: it yields all currently buffered
        events and then exits.  For a live stream, the caller should loop
        while repeatedly calling this (or use the server's ``/feed`` endpoint
        which handles long-polling).

        Parameters
        ----------
        poll_interval:
            Unused in snapshot mode; reserved for future streaming extension.
        max_events:
            Maximum number of events to yield.  ``None`` means all buffered.

        Yields
        ------
        str
            SSE-formatted event string (``data: {...}\\n\\n``).
        """
        with self._lock:
            events = list(self._events)
        if max_events is not None:
            events = events[-max_events:]
        for event in events:
            yield event.to_sse()

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of feed activity.

        Returns
        -------
        dict
            Keys: ``total_events``, ``critical``, ``high``, ``medium``, ``low``.
        """
        with self._lock:
            total = len(self._events)
            by_sev: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for e in self._events:
                if e.severity in by_sev:
                    by_sev[e.severity] += 1
        return {
            "total_events": total,
            **by_sev,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store(self, payload: ThreatIntelPayload) -> None:
        """Thread-safely append *payload* and enforce the ring-buffer limit."""
        with self._lock:
            self._events.append(payload)
            if len(self._events) > self.max_events:
                # Trim oldest entries
                overflow = len(self._events) - self.max_events
                self._events = self._events[overflow:]
