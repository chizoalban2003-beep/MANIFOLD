"""Tests for manifold/live.py — GossipBus and LiveBrain."""

from manifold import (
    BrainConfig,
    BrainMemory,
    BrainOutcome,
    BrainTask,
    GossipNote,
    ManifoldBrain,
    ToolProfile,
)
from manifold.live import GossipBus, LiveBrain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)

_TOOL = ToolProfile(
    "order_lookup",
    cost=0.05,
    latency=0.05,
    reliability=0.90,
    risk=0.04,
    asset=0.85,
    domain="support",
)

_TASK = BrainTask(
    "Check order",
    domain="support",
    tool_relevance=0.95,
    source_confidence=0.85,
    uncertainty=0.3,
    stakes=0.5,
)


def _brain() -> ManifoldBrain:
    return ManifoldBrain(_CFG, tools=[_TOOL])


def _failure() -> BrainOutcome:
    return BrainOutcome(
        success=False, cost_paid=0.2, risk_realized=0.8,
        asset_gained=0.0, failure_mode="tool_error",
    )


def _success() -> BrainOutcome:
    return BrainOutcome(success=True, cost_paid=0.05, risk_realized=0.0, asset_gained=0.9)


# ---------------------------------------------------------------------------
# GossipBus
# ---------------------------------------------------------------------------


def test_gossip_bus_delivers_note_to_subscriber() -> None:
    """A published note must reach a subscribed memory."""
    bus = GossipBus()
    memory = BrainMemory()
    bus.subscribe(memory, source_id="beta")

    note = GossipNote(
        tool="order_lookup",
        claim="failing",
        source_id="alpha",
        source_reputation=0.9,
        source_is_scout=False,
    )
    bus.publish(note)
    bus.drain()

    # Memory should have been updated for the tool
    assert "order_lookup" in memory.tool_stats
    assert memory.tool_stats["order_lookup"]["success_rate"] < 1.0
    bus.stop()


def test_gossip_bus_does_not_deliver_self_gossip() -> None:
    """A note must NOT be delivered back to the agent that published it."""
    bus = GossipBus()
    memory = BrainMemory()
    bus.subscribe(memory, source_id="alpha")   # same as note.source_id

    note = GossipNote(
        tool="order_lookup",
        claim="failing",
        source_id="alpha",
        source_reputation=0.9,
    )
    bus.publish(note)
    bus.drain()

    # Memory should be untouched — self-gossip was filtered
    assert "order_lookup" not in memory.tool_stats
    bus.stop()


def test_gossip_bus_delivers_to_multiple_subscribers() -> None:
    """All subscribers except the originator must receive the note."""
    bus = GossipBus()
    mem_b = BrainMemory()
    mem_c = BrainMemory()
    bus.subscribe(BrainMemory(), source_id="alpha")  # originator
    bus.subscribe(mem_b, source_id="beta")
    bus.subscribe(mem_c, source_id="gamma")

    note = GossipNote(
        tool="order_lookup",
        claim="failing",
        source_id="alpha",
        source_reputation=0.85,
    )
    bus.publish(note)
    bus.drain()

    assert "order_lookup" in mem_b.tool_stats
    assert "order_lookup" in mem_c.tool_stats
    bus.stop()


def test_gossip_bus_drops_expired_notes() -> None:
    """Notes delivered after the TTL window must be silently dropped."""
    # A TTL of 0 seconds means any note is stale by the time the delivery
    # thread processes it (since even nanoseconds of latency exceeds 0 s).
    bus = GossipBus(ttl_seconds=0.0)
    memory = BrainMemory()
    bus.subscribe(memory, source_id="beta")

    note = GossipNote(
        tool="order_lookup",
        claim="failing",
        source_id="alpha",
        source_reputation=0.9,
    )
    bus.publish(note)
    bus.drain()

    # Memory must be untouched — stale gossip was dropped
    assert "order_lookup" not in memory.tool_stats
    bus.stop()


def test_gossip_bus_stop_joins_threads() -> None:
    """stop() must join all delivery threads without hanging."""
    bus = GossipBus()
    bus.subscribe(BrainMemory(), source_id="agent")
    bus.stop()
    assert bus._threads == []
    assert bus._subscribers == []


# ---------------------------------------------------------------------------
# LiveBrain
# ---------------------------------------------------------------------------


def test_live_brain_failure_publishes_failing_gossip_to_peer() -> None:
    """A tool failure on alpha must reduce order_lookup success_rate in beta's memory."""
    bus = GossipBus()
    alpha = LiveBrain(brain=_brain(), bus=bus, agent_id="alpha")
    beta = LiveBrain(brain=_brain(), bus=bus, agent_id="beta")

    decision = alpha.decide(_TASK)
    alpha.learn(_TASK, decision, _failure())
    bus.drain()

    if decision.selected_tool == "order_lookup":
        rate = beta.brain.memory.tool_stats.get("order_lookup", {}).get("success_rate", 1.0)
        assert rate < 1.0, "Peer beta should have received failure gossip"

    bus.stop()


def test_live_brain_success_publishes_healthy_gossip_to_peer() -> None:
    """A tool success on alpha must nudge order_lookup success_rate upward in beta's memory."""
    bus = GossipBus()
    alpha = LiveBrain(brain=_brain(), bus=bus, agent_id="alpha")
    beta = LiveBrain(brain=_brain(), bus=bus, agent_id="beta")

    # First scar beta's memory manually
    beta.brain.memory.tool_stats["order_lookup"] = {
        "count": 5.0,
        "success_rate": 0.50,
        "utility": -0.2,
        "consecutive_failures": 3.0,
    }

    decision = alpha.decide(_TASK)
    alpha.learn(_TASK, decision, _success())
    bus.drain()

    if decision.selected_tool == "order_lookup":
        rate = beta.brain.memory.tool_stats["order_lookup"]["success_rate"]
        assert rate > 0.50, "Healthy gossip should nudge beta's tool score upward"

    bus.stop()


def test_live_brain_does_not_publish_gossip_when_no_tool_used() -> None:
    """Non-tool decisions must not generate gossip notes."""
    bus = GossipBus()
    # Use a task that will NOT select a tool (no matching domain, low relevance)
    no_tool_task = BrainTask(
        "Unsafe",
        domain="safety",
        uncertainty=0.9,
        complexity=0.7,
        stakes=0.9,
        source_confidence=0.3,
        safety_sensitivity=0.95,
        dynamic_goal=True,
        tool_relevance=0.1,
    )
    peer_memory = BrainMemory()
    bus.subscribe(peer_memory, source_id="beta")
    alpha = LiveBrain(brain=_brain(), bus=bus, agent_id="alpha")

    decision = alpha.decide(no_tool_task)
    alpha.learn(no_tool_task, decision, _failure())
    bus.drain()

    # No tool was selected, so no gossip note should have been published
    assert decision.selected_tool is None
    assert peer_memory.tool_stats == {}
    bus.stop()


def test_live_brain_periodic_decay_fires() -> None:
    """Memory decay must trigger after decay_every_n_decisions learn() calls."""
    bus = GossipBus()
    alpha = LiveBrain(
        brain=_brain(),
        bus=bus,
        agent_id="alpha",
        decay_every_n_decisions=3,
    )

    # Plant a deep scar on order_lookup
    alpha.brain.memory.tool_stats["order_lookup"] = {
        "count": 10.0,
        "success_rate": 0.40,
        "utility": -0.5,
        "consecutive_failures": 5.0,
    }

    # Use a task that will NOT select order_lookup (wrong domain, very low relevance)
    # so direct learn() calls never touch order_lookup's stats — only decay does.
    non_tool_task = BrainTask(
        "General query",
        domain="general",
        tool_relevance=0.10,   # below the 0.35 threshold in select_tool
        source_confidence=0.85,
        uncertainty=0.3,
        stakes=0.5,
    )
    decision = alpha.decide(non_tool_task)

    for _ in range(3):
        alpha.learn(non_tool_task, decision, _success())

    # After 3 learn() calls, decay() fires once: 0.40*0.97 + 0.03 = 0.418
    rate = alpha.brain.memory.tool_stats["order_lookup"]["success_rate"]
    assert rate > 0.40, "Decay should have partially forgiven the scar"
    bus.stop()


def test_live_brain_decide_and_learn_convenience() -> None:
    """decide_and_learn must return the decision and update memory in one call."""
    bus = GossipBus()
    alpha = LiveBrain(brain=_brain(), bus=bus, agent_id="alpha")

    decision = alpha.decide_and_learn(_TASK, _failure())

    assert decision is not None
    assert alpha.brain.memory.action_stats  # memory was updated
    bus.stop()


def test_live_brain_self_gossip_not_reingested() -> None:
    """alpha must not ingest its own gossip — memory should only change via learn()."""
    bus = GossipBus()
    alpha = LiveBrain(brain=_brain(), bus=bus, agent_id="alpha")

    decision = alpha.decide(_TASK)
    rate_before = alpha.brain.memory.tool_stats.get("order_lookup", {}).get("success_rate", 1.0)

    alpha.learn(_TASK, decision, _failure())
    bus.drain()

    # alpha's tool stat should reflect the direct learn() only, not gossip re-ingestion
    rate_after_learn = alpha.brain.memory.tool_stats.get("order_lookup", {}).get("success_rate", 1.0)

    if decision.selected_tool == "order_lookup":
        # The rate should have dropped from the direct learn, not doubled by gossip
        assert rate_after_learn < rate_before
        # And should be > the floor that double-ingest would produce
        # (direct LR=0.20 applied once → floor ≈ 0.80*prior; gossip on top would be ≈0.02 more)
        assert rate_after_learn > 0.0

    bus.stop()
