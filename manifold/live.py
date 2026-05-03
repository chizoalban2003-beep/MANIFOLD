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
    BrainDecision,
    BrainMemory,
    BrainOutcome,
    BrainTask,
    DecompositionPlan,
    GossipNote,
    HierarchicalBrain,
    HierarchicalDecision,
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
                age_seconds = time.monotonic() - envelope.published_at
                if age_seconds > self.ttl_seconds:
                    continue  # perishable note expired in transit; drop it
                aged_note = dataclasses.replace(envelope.note, age_minutes=age_seconds / 60.0)
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


# ---------------------------------------------------------------------------
# Hierarchical live execution: child brains wired to global GossipBus
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalLiveBrain(HierarchicalBrain):
    """``HierarchicalBrain`` with child brains wired to the global ``GossipBus``.

    When ``HierarchicalBrain`` decomposes a complex task, each child brain is
    wrapped as a ``LiveBrain`` on the **same** ``GossipBus`` as the parent.
    This means:

    * A tool failure discovered by a ``[Research]`` child is broadcast to the
      *entire ecosystem* — all ``LiveBrain`` peers that share the bus receive
      the ``"failing"`` gossip note and update their tool memory.
    * A successful child outcome broadcasts a ``"healthy"`` note, letting the
      ecosystem recover faster than passive memory decay alone.
    * Self-gossip prevention is maintained: each child has its own ``agent_id``
      (``"{parent_id}/child_{i}"``), so it does not receive its own note back.

    The parent brain itself is also subscribed to the bus (if *agent_id* is
    given), so it too receives child failure gossip and updates accordingly.

    Parameters
    ----------
    bus:
        The shared ``GossipBus`` instance.  If ``None``, the brain falls back
        to standard (non-gossip) ``HierarchicalBrain`` behaviour.
    agent_id:
        Identifier for this parent brain on the bus.  Also used as the prefix
        for child agent IDs: ``"{agent_id}/child_0"``, ``"{agent_id}/child_1"``, …
    agent_reputation:
        Reputation attached to outgoing gossip notes.  Defaults to 0.8.
    decay_every_n_decisions:
        Parent brain memory decay cadence (decisions between decay ticks).
        Defaults to 20.  Set to 0 to disable.

    Example
    -------
    ::

        bus = GossipBus()
        brain_a = HierarchicalLiveBrain(config, tools=tools, bus=bus, agent_id="a")
        brain_b = LiveBrain(ManifoldBrain(config, tools=tools), bus=bus, agent_id="b")

        hd = brain_a.decide_hierarchical(task)
        # If decomposed, child brains have already published gossip to "b".
        # brain_b learns about tool failures without being involved in the task.
        bus.drain()
    """

    bus: GossipBus | None = None
    agent_id: str = ""
    agent_reputation: float = 0.8
    decay_every_n_decisions: int = 20
    _decision_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bus and self.agent_id:
            self.bus.subscribe(self.memory, source_id=self.agent_id)

    def decide_hierarchical(
        self, task: BrainTask, depth: int = 0
    ) -> HierarchicalDecision:
        """Decide and, if decomposed, wire all child brains to the global bus.

        Overrides ``HierarchicalBrain.decide_hierarchical`` to inject bus
        wiring into child ``LiveBrain`` instances produced during decomposition.
        When ``bus`` is ``None`` (no bus configured), delegates directly to the
        parent implementation without modification.

        Parameters
        ----------
        task:
            The task to route.
        depth:
            Current recursion depth (0 for the root call).
        """
        if self.bus is None:
            return super().decide_hierarchical(task, depth)

        # Delegate the structural decision to the parent implementation first.
        hd = super().decide_hierarchical(task, depth)
        # We already published child gossip in _execute_sub_tasks_live, but for
        # the non-decomposed path nothing extra is needed.
        return hd

    def _execute_sub_tasks(  # type: ignore[override]
        self, plan: DecompositionPlan, parent_task: BrainTask
    ) -> tuple[BrainDecision, ...]:
        """Create child ``LiveBrain`` instances on the global bus and run decisions.

        Each child:
        * Is wrapped as a ``LiveBrain`` with a unique ID ``"{agent_id}/child_{i}"``.
        * Subscribes to ``self.bus`` so its gossip is heard by all peers.
        * On a tool-using decision, the child's ``_publish_tool_gossip`` fires
          immediately after the simulated outcome (success assumed for pure
          ``decide()`` — real outcomes are published when ``learn_child()`` is
          called externally).

        For the pure ``decide_hierarchical`` path (no real outcome yet), no
        gossip is published by children because we don't have an outcome.
        Gossip is only published when ``learn_child`` is called with a real
        ``BrainOutcome``.
        """
        if self.bus is None:
            return super()._execute_sub_tasks(plan, parent_task)

        decisions = []
        for i, st in enumerate(plan.sub_tasks):
            child_task = BrainTask(
                prompt=st.prompt,
                domain=st.domain,
                complexity=st.complexity,
                stakes=st.stakes,
                uncertainty=st.uncertainty,
                source_confidence=st.source_confidence,
                tool_relevance=parent_task.tool_relevance,
                time_pressure=parent_task.time_pressure,
                safety_sensitivity=parent_task.safety_sensitivity,
                collaboration_value=parent_task.collaboration_value,
                user_patience=parent_task.user_patience,
            )
            child_id = f"{self.agent_id}/child_{i}" if self.agent_id else f"child_{i}"
            child_brain = ManifoldBrain(
                config=self.config,
                tools=self.tools,
                memory=BrainMemory(),  # intentional: child brains start with clean memory
                price_adapter=self.price_adapter,
            )
            child_live = LiveBrain(
                brain=child_brain,
                bus=self.bus,
                agent_id=child_id,
                agent_reputation=self.agent_reputation,
            )
            decisions.append(child_live.decide(child_task))
        return tuple(decisions)

    def learn_child(
        self,
        child_index: int,
        task: BrainTask,
        decision: BrainDecision,
        outcome: BrainOutcome,
    ) -> None:
        """Publish gossip for a child brain outcome to the global bus.

        After a child brain's sub-task completes in the real world (i.e. an
        actual tool outcome is known), call this method to broadcast the result.
        This is the mechanism by which a child failure (e.g. hallucinating tool)
        propagates to the entire ecosystem.

        Parameters
        ----------
        child_index:
            Index of the child (0-based) within the decomposition plan.
        task:
            The child's ``BrainTask``.
        decision:
            The child's ``BrainDecision``.
        outcome:
            The real-world outcome for the child's action.
        """
        if self.bus is None or not decision.selected_tool:
            return
        child_id = f"{self.agent_id}/child_{child_index}" if self.agent_id else f"child_{child_index}"
        claim = "failing" if not outcome.success else "healthy"
        note = GossipNote(
            tool=decision.selected_tool,
            claim=claim,
            source_id=child_id,
            source_reputation=self.agent_reputation,
            source_is_scout=False,
            confidence=1.0,
            age_minutes=0.0,
        )
        self.bus.publish(note)

    def learn(
        self,
        task: BrainTask,
        decision: BrainDecision,
        outcome: BrainOutcome,
    ) -> None:
        """Update parent memory and publish tool gossip for monolithic decisions.

        For decisions that were NOT decomposed (i.e. the parent acted directly),
        this mirrors ``LiveBrain.learn`` — updating memory and broadcasting
        tool gossip to the ecosystem.  For decomposed decisions, call
        ``learn_child()`` per sub-task as outcomes arrive.
        """
        super().learn(task, decision, outcome)
        if self.bus and decision.selected_tool:
            claim = "failing" if not outcome.success else "healthy"
            note = GossipNote(
                tool=decision.selected_tool,
                claim=claim,
                source_id=self.agent_id or "hierarchical",
                source_reputation=self.agent_reputation,
                source_is_scout=False,
                confidence=1.0,
                age_minutes=0.0,
            )
            self.bus.publish(note)
        self._decision_count += 1
        if (
            self.decay_every_n_decisions > 0
            and self._decision_count % self.decay_every_n_decisions == 0
        ):
            self.memory.decay()
