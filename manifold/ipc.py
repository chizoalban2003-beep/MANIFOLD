"""Phase 49: IPC Event Bus — Pub/Sub Architecture for the MANIFOLD Daemon.

Replaces tight coupling between MANIFOLD subsystems with a non-blocking
Publisher/Subscriber architecture.  All internal events flow through the
:class:`EventBus`, decoupling producers (server, sandbox, sharding) from
consumers (shell TUI, dashboard, logging).

Architecture
------------
* **Thread-safe** — a :class:`threading.Condition` (backed by a
  :class:`threading.Lock`) protects the subscriber registry and signals
  waiting consumers.
* **Non-blocking publish** — :meth:`EventBus.publish` enqueues events on
  subscriber :class:`queue.Queue` objects using ``put_nowait``.  Slow
  consumers do not block the publisher; a ``queue.Full`` exception is
  silently swallowed (the event is dropped for that subscriber).
* **Topic routing** — topics are dotted-path strings (e.g.
  ``"sandbox.violation"``, ``"dht.peer.joined"``).  Subscribers can use
  exact match or wildcard patterns (``"sandbox.*"``).

Standard topics
---------------
``system.entropy.high``
    System entropy has crossed a warning threshold.
``dht.peer.joined``
    A new peer joined the DHT routing table.
``dht.peer.dropped``
    A peer was evicted from the DHT routing table.
``sandbox.violation``
    The AST validator rejected a code submission.
``sandbox.timeout``
    A sandboxed execution exceeded its opcode budget.
``meta.champion.promoted``
    A Challenger prompt was promoted to Champion.
``vector.entry.added``
    A vector was added to the :class:`~manifold.vectorfs.VectorIndex`.
``admin.veto``
    An admin manually forced a tool's reputation to zero.

Key classes
-----------
``Event``
    Immutable event envelope with a topic, payload dict, and timestamp.
``EventBus``
    Thread-safe pub/sub broker.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """An immutable event envelope published to the :class:`EventBus`.

    Attributes
    ----------
    topic:
        Dotted-path string identifying the event type (e.g.
        ``"sandbox.violation"``).
    payload:
        Arbitrary key/value data accompanying the event.
    timestamp:
        POSIX timestamp when the event was created.
    """

    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "topic": self.topic,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Topic pattern matching
# ---------------------------------------------------------------------------


def _topic_matches(topic: str, pattern: str) -> bool:
    """Return ``True`` if *topic* matches *pattern*.

    Matching rules
    --------------
    * ``"*"`` — matches everything.
    * ``"prefix.*"`` — matches *prefix* itself **and** any
      dotted sub-topic of *prefix* (e.g. ``"sandbox.*"`` matches
      ``"sandbox.violation"`` and ``"sandbox.timeout"``).
    * Any other string — exact match only.
    """
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return topic == prefix or topic.startswith(prefix + ".")
    return topic == pattern


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


@dataclass
class EventBus:
    """Thread-safe topic-based pub/sub broker.

    Parameters
    ----------
    max_queue_size:
        Maximum number of events each subscriber queue can buffer before
        new events are dropped.  Default: ``1_000``.

    Example
    -------
    ::

        bus = EventBus()
        q = bus.subscribe("sandbox.*")
        bus.publish("sandbox.violation", {"source_hash": "abc123"})
        event = q.get_nowait()
        assert event.topic == "sandbox.violation"
    """

    max_queue_size: int = 1_000

    # internal state (not part of public API)
    _subscribers: dict[str, list[queue.Queue[Event]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, pattern: str) -> "queue.Queue[Event]":
        """Subscribe to events matching *pattern*.

        Parameters
        ----------
        pattern:
            Topic pattern to subscribe to.  Supports exact match and
            wildcard suffix (``"sandbox.*"``).

        Returns
        -------
        queue.Queue[Event]
            A FIFO queue that receives matching events.  Call
            :meth:`queue.Queue.get` or :meth:`queue.Queue.get_nowait`
            to consume events.
        """
        q: queue.Queue[Event] = queue.Queue(maxsize=self.max_queue_size)
        with self._lock:
            self._subscribers.setdefault(pattern, []).append(q)
        return q

    def unsubscribe(self, pattern: str, q: "queue.Queue[Event]") -> None:
        """Remove *q* from the subscriber list for *pattern*.

        Parameters
        ----------
        pattern:
            The pattern that was used when subscribing.
        q:
            The queue returned by :meth:`subscribe`.
        """
        with self._lock:
            subs = self._subscribers.get(pattern, [])
            try:
                subs.remove(q)
            except ValueError:
                pass

    def subscriber_count(self, pattern: str | None = None) -> int:
        """Return the number of subscriber queues.

        Parameters
        ----------
        pattern:
            If given, count only subscribers for this exact pattern.
            If ``None``, return the total count across all patterns.
        """
        with self._lock:
            if pattern is not None:
                return len(self._subscribers.get(pattern, []))
            return sum(len(qs) for qs in self._subscribers.values())

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Publish an event on *topic*.

        Matching is done by comparing *topic* against every registered
        subscription pattern using :func:`_topic_matches`.

        Parameters
        ----------
        topic:
            Dotted-path event topic (e.g. ``"sandbox.violation"``).
        payload:
            Optional key/value data to include in the event.

        Returns
        -------
        int
            Number of subscriber queues the event was delivered to (queues
            that were full at delivery time are excluded from the count).
        """
        event = Event(topic=topic, payload=payload or {})
        delivered = 0

        with self._condition:
            for sub_pattern, queues in list(self._subscribers.items()):
                if _topic_matches(topic, sub_pattern):
                    for q in queues:
                        try:
                            q.put_nowait(event)
                            delivered += 1
                        except queue.Full:
                            pass
            self._condition.notify_all()

        return delivered

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def wait_for_event(
        self,
        q: "queue.Queue[Event]",
        *,
        timeout: float = 1.0,
    ) -> Event | None:
        """Block until an event arrives on *q* or *timeout* seconds pass.

        Parameters
        ----------
        q:
            A queue returned by :meth:`subscribe`.
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        Event | None
            The first event from *q*, or ``None`` if the timeout expired.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return q.get_nowait()
            except queue.Empty:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            with self._condition:
                self._condition.wait(timeout=min(remaining, 0.05))

    def registered_patterns(self) -> list[str]:
        """Return all currently registered subscription patterns."""
        with self._lock:
            return list(self._subscribers.keys())


# ---------------------------------------------------------------------------
# Standard topic constants
# ---------------------------------------------------------------------------

TOPIC_SYSTEM_ENTROPY_HIGH = "system.entropy.high"
TOPIC_DHT_PEER_JOINED = "dht.peer.joined"
TOPIC_DHT_PEER_DROPPED = "dht.peer.dropped"
TOPIC_SANDBOX_VIOLATION = "sandbox.violation"
TOPIC_SANDBOX_TIMEOUT = "sandbox.timeout"
TOPIC_META_CHAMPION_PROMOTED = "meta.champion.promoted"
TOPIC_VECTOR_ENTRY_ADDED = "vector.entry.added"
TOPIC_ADMIN_VETO = "admin.veto"
