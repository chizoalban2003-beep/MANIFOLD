"""Live execution layer for MANIFOLD Brain.

Provides two components that bridge the validated gossip-weighting module to a
running multi-agent loop:

GossipBus
    An in-process async broadcast bus built on stdlib ``threading`` and
    ``queue``.  Each subscribed ``BrainMemory`` gets its own delivery queue
    served by a daemon thread.  ``publish()`` is fire-and-forget; the deciding
    agent never blocks waiting for peers to receive the note.  Note age is
    computed at *delivery* time so ``0.97^age`` decay in ``ingest_gossip``
    reflects true propagation latency naturally.

LiveBrain
    A thin wrapper around ``ManifoldBrain`` that integrates automatic gossip
    publication after tool outcomes and an optional periodic memory decay tick.
    ``decide()`` never touches the bus — it reads only local memory and is
    therefore always O(1) with no network dependency.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from dataclasses import dataclass, field

from .brain import (
    BrainConfig,
    BrainDecision,
    BrainMemory,
    BrainOutcome,
    BrainTask,
    GossipNote,
    ManifoldBrain,
)


# Sentinel object used to signal delivery threads to stop cleanly.
_STOP = object()


@dataclass
class _GossipEnvelope:
    """Internal envelope that records the monotonic time a note was published."""

    note: GossipNote
    published_at: float = field(default_factory=time.monotonic)


@dataclass
class GossipBus:
    """In-process async epidemic broadcast bus for ``GossipNote`` instances.

    Notes are delivered asynchronously to all subscribed ``BrainMemory``
    instances *except* the one that published the note (self-gossip loop
    prevention).  Age is stamped at delivery time, so propagation latency
    contributes to the ``0.97^age`` weight decay already implemented in
    ``BrainMemory.ingest_gossip``.

    The bus is designed for single-process use (simulations, unit tests,
    single-host deployments).  The interface is intentionally kept thin so
    the transport layer can be swapped to Redis pub/sub or NATS by replacing
    ``publish`` and ``subscribe`` without changing ``BrainMemory`` or
    ``LiveBrain``.

    Parameters
    ----------
    ttl_seconds:
        Notes older than this value are silently dropped during delivery.
        Defaults to 300 s (5 minutes), matching the ``0.97^age`` decay window.

    Example
    -------
    ::

        bus = GossipBus()
        bus.subscribe(brain_a.memory, source_id="agent_a")
        bus.subscribe(brain_b.memory, source_id="agent_b")

        bus.publish(GossipNote(
            tool="database", claim="failing",
            source_id="agent_a", source_reputation=0.8,
        ))

        bus.drain()   # wait for async delivery (useful in tests)
        bus.stop()
    """

    ttl_seconds: float = 300.0

    _subscribers: list[tuple[str, BrainMemory, queue.Queue]] = field(
        default_factory=list, init=False, repr=False
    )
    _threads: list[threading.Thread] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def subscribe(self, memory: BrainMemory, source_id: str = "") -> None:
        """Register *memory* to receive notes published on this bus.

        Parameters
        ----------
        memory:
            The ``BrainMemory`` instance that will receive gossip.
        source_id:
            The agent identifier associated with *memory*.  Notes whose
            ``source_id`` matches this value are *not* delivered back, which
            prevents an agent from reinforcing its own gossip.
        """
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append((source_id, memory, q))
        t = threading.Thread(
            target=self._delivery_loop,
            args=(memory, q),
            daemon=True,
            name=f"gossip-{source_id or id(memory)}",
        )
        t.start()
        with self._lock:
            self._threads.append(t)

    def publish(self, note: GossipNote) -> None:
        """Broadcast *note* to all subscribers except the originator.

        Non-blocking — returns immediately.  Each subscriber queue is
        thread-safe; concurrent publishers are fully supported.
        """
        envelope = _GossipEnvelope(note=note)
        with self._lock:
            snapshot = list(self._subscribers)
        for sub_id, _memory, q in snapshot:
            if sub_id != note.source_id:
                q.put_nowait(envelope)

    def drain(self) -> None:
        """Block until every queued note has been delivered to all subscribers.

        Useful in tests and simulation steps where you need to assert on
        memory state immediately after publishing.
        """
        with self._lock:
            queues = [q for _, _, q in self._subscribers]
        for q in queues:
            q.join()

    def stop(self) -> None:
        """Signal all delivery threads to exit and wait for them to finish."""
        with self._lock:
            snapshot = list(self._subscribers)
            threads = list(self._threads)
        for _sub_id, _memory, q in snapshot:
            q.put_nowait(_STOP)
        for t in threads:
            t.join(timeout=2.0)
        with self._lock:
            self._subscribers.clear()
            self._threads.clear()

    def _delivery_loop(self, memory: BrainMemory, q: queue.Queue) -> None:
        while True:
            item = q.get()
            try:
                if item is _STOP:
                    return
                envelope: _GossipEnvelope = item
                age_minutes = (time.monotonic() - envelope.published_at) / 60.0
                if age_minutes * 60.0 > self.ttl_seconds:
                    continue  # perishable note expired in transit; drop it
                aged_note = dataclasses.replace(envelope.note, age_minutes=age_minutes)
                memory.ingest_gossip(aged_note)
            finally:
                q.task_done()


@dataclass
class LiveBrain:
    """``ManifoldBrain`` with integrated gossip publication and periodic decay.

    After every ``learn()`` call involving a tool, ``LiveBrain`` automatically
    constructs a ``GossipNote`` reflecting the outcome and publishes it to
    *bus*.  Peers that share the same ``GossipBus`` will receive the note
    asynchronously and update their own tool memory — without the publishing
    agent waiting for confirmation.

    ``decide()`` is a pure delegation to the underlying brain and never touches
    the bus.  It reads only local memory, so it is always instant.

    Parameters
    ----------
    brain:
        Underlying ``ManifoldBrain`` instance.
    bus:
        ``GossipBus`` that failure and recovery notes are published to.
    agent_id:
        Unique identifier for this agent.  Used as ``source_id`` on gossip
        notes and as the filter key to prevent self-delivery in the bus.
    agent_reputation:
        Reputation score attached to outgoing ``GossipNote`` instances [0, 1].
        Defaults to 0.8.
    decay_every_n_decisions:
        Call ``brain.memory.decay()`` every *N* completed ``learn()`` calls.
        Defaults to 20.  Set to 0 to disable automatic decay.

    Example
    -------
    ::

        bus = GossipBus()

        alpha = LiveBrain(
            brain=ManifoldBrain(BrainConfig(), tools=default_tools()),
            bus=bus,
            agent_id="alpha",
        )
        beta = LiveBrain(
            brain=ManifoldBrain(BrainConfig(), tools=default_tools()),
            bus=bus,
            agent_id="beta",
        )

        task = BrainTask("lookup", domain="support", tool_relevance=0.9)
        outcome = BrainOutcome(success=False, cost_paid=0.2,
                               risk_realized=0.8, asset_gained=0.0,
                               failure_mode="tool_error")

        decision = alpha.decide(task)
        alpha.learn(task, decision, outcome)   # publishes failure gossip to beta
        bus.drain()                            # ensure beta received it
    """

    brain: ManifoldBrain
    bus: GossipBus
    agent_id: str
    agent_reputation: float = 0.8
    decay_every_n_decisions: int = 20

    _decision_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.bus.subscribe(self.brain.memory, source_id=self.agent_id)

    def decide(self, task: BrainTask) -> BrainDecision:
        """Return the brain's decision — instant, never touches the bus."""
        return self.brain.decide(task)

    def learn(
        self,
        task: BrainTask,
        decision: BrainDecision,
        outcome: BrainOutcome,
    ) -> None:
        """Update local memory and asynchronously broadcast a gossip note.

        For every tool-using decision:

        * A ``"failing"`` note is published on failure.
        * A ``"healthy"`` note is published on success (acts as a "social
          band-aid" that lets peers forgive a tool faster than decay alone).

        The decay tick fires every ``decay_every_n_decisions`` calls.
        """
        self.brain.learn(task, decision, outcome)
        self._publish_tool_gossip(decision, outcome)
        self._decision_count += 1
        if (
            self.decay_every_n_decisions > 0
            and self._decision_count % self.decay_every_n_decisions == 0
        ):
            self.brain.memory.decay()

    def decide_and_learn(
        self,
        task: BrainTask,
        outcome: BrainOutcome,
    ) -> BrainDecision:
        """Convenience: decide, immediately record the outcome, return the decision.

        Useful in simulation loops where the outcome is known synchronously.
        """
        decision = self.decide(task)
        self.learn(task, decision, outcome)
        return decision

    def _publish_tool_gossip(
        self,
        decision: BrainDecision,
        outcome: BrainOutcome,
    ) -> None:
        if not decision.selected_tool:
            return
        claim = "failing" if not outcome.success else "healthy"
        note = GossipNote(
            tool=decision.selected_tool,
            claim=claim,
            source_id=self.agent_id,
            source_reputation=self.agent_reputation,
            source_is_scout=False,
            confidence=1.0,
            age_minutes=0.0,
        )
        self.bus.publish(note)
