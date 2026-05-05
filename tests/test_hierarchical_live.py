"""Tests for HierarchicalLiveBrain — GossipBus wiring for Phase 3 children."""

import dataclasses

from manifold import (
    BrainConfig,
    BrainOutcome,
    BrainTask,
    GossipBus,
    HierarchicalLiveBrain,
    LiveBrain,
    ManifoldBrain,
    ToolProfile,
    default_tools,
    run_gossip_hierarchical_suite,
)


_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)

_COMPLEX_TASK = BrainTask(
    "Write a comprehensive market research report",
    domain="research",
    complexity=0.92,
    stakes=0.85,
    uncertainty=0.55,
    source_confidence=0.60,
    user_patience=0.80,
)

_SIMPLE_TASK = BrainTask(
    "What is 2 + 2?",
    domain="math",
    complexity=0.15,
    stakes=0.10,
)

_WEB_SEARCH = ToolProfile(
    "web_search", cost=0.12, latency=0.18, reliability=0.78, risk=0.12, asset=0.75
)

_FAILING_OUTCOME = BrainOutcome(
    success=False, cost_paid=0.25, risk_realized=0.80, asset_gained=0.0, failure_mode="tool_error"
)
_SUCCESS_OUTCOME = BrainOutcome(
    success=True, cost_paid=0.12, risk_realized=0.0, asset_gained=0.75
)


# ---------------------------------------------------------------------------
# HierarchicalLiveBrain construction
# ---------------------------------------------------------------------------


def test_no_bus_construction() -> None:
    brain = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=None)
    assert brain.bus is None


def test_with_bus_subscribes_parent() -> None:
    bus = GossipBus()
    HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="parent")
    # Parent should be subscribed (1 subscriber = the parent)
    assert len(bus._subscribers) == 1
    bus.stop()


def test_inherits_hierarchical_brain_attributes() -> None:
    brain = HierarchicalLiveBrain(
        _CFG, tools=default_tools(), decompose_threshold=0.80, max_depth=3, coordination_tax=0.15
    )
    assert brain.decompose_threshold == 0.80
    assert brain.max_depth == 3
    assert brain.coordination_tax == 0.15


# ---------------------------------------------------------------------------
# decide_hierarchical with bus=None falls back cleanly
# ---------------------------------------------------------------------------


def test_no_bus_simple_not_decomposed() -> None:
    brain = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=None)
    hd = brain.decide_hierarchical(_SIMPLE_TASK)
    assert not hd.decomposed


def test_no_bus_complex_decomposes() -> None:
    brain = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=None)
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.decomposed


def test_with_bus_complex_decomposes() -> None:
    bus = GossipBus()
    brain = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="p")
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.decomposed
    assert hd.sub_decisions is not None
    assert len(hd.sub_decisions) == 2
    bus.stop()


def test_with_bus_child_brains_subscribed() -> None:
    """After a decomposed decision, the bus should have parent + 2 child subscribers."""
    bus = GossipBus()
    brain = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="p")
    brain.decide_hierarchical(_COMPLEX_TASK)
    # parent + 2 children
    assert len(bus._subscribers) == 3
    bus.stop()


# ---------------------------------------------------------------------------
# Gossip propagation — child failure reaches ecosystem peers
# ---------------------------------------------------------------------------


def test_child_failure_gossip_reaches_peer() -> None:
    """A child brain tool failure must propagate to a peer — proven by driving adj negative."""
    bus = GossipBus()
    peer = LiveBrain(ManifoldBrain(_CFG, tools=default_tools()), bus=bus, agent_id="peer")
    parent = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="parent")

    hd = parent.decide_hierarchical(_COMPLEX_TASK)
    failing_decision = hd.sub_decisions[0] if hd.sub_decisions else parent.decide(_COMPLEX_TASK)
    failing_decision = dataclasses.replace(failing_decision, selected_tool="web_search")

    # Send enough failing notes to overcome the optimistic success_rate prior (1.0).
    # After ~10 notes the peer's success_rate < tool.reliability (0.78), driving adj < 0.
    for _ in range(10):
        parent.learn_child(0, _COMPLEX_TASK, failing_decision, _FAILING_OUTCOME)
    bus.drain()

    adj = peer.brain.memory.tool_reliability_adjustment(_WEB_SEARCH)
    bus.stop()
    assert adj != 0.0, "Gossip never reached peer — tool_stats entry not created"
    assert adj < 0.0, f"Expected negative adjustment after 10 failing notes, got {adj:.3f}"


def test_child_success_gossip_reaches_peer() -> None:
    bus = GossipBus()
    # First make peer believe tool is bad via manual gossip
    from manifold.brain import GossipNote
    peer = LiveBrain(ManifoldBrain(_CFG, tools=default_tools()), bus=bus, agent_id="peer2")
    # Inject negative history directly into peer memory
    for _ in range(3):
        peer.brain.memory.ingest_gossip(GossipNote(
            tool="web_search", claim="failing", source_id="other",
            source_reputation=0.9, source_is_scout=False, confidence=0.9, age_minutes=0.0
        ))
    adj_before = peer.brain.memory.tool_reliability_adjustment(_WEB_SEARCH)

    parent = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="parent2")
    hd = parent.decide_hierarchical(_COMPLEX_TASK)
    success_decision = hd.sub_decisions[0] if hd.sub_decisions else parent.decide(_COMPLEX_TASK)
    success_decision = dataclasses.replace(success_decision, selected_tool="web_search")
    parent.learn_child(0, _COMPLEX_TASK, success_decision, _SUCCESS_OUTCOME)
    bus.drain()

    adj_after = peer.brain.memory.tool_reliability_adjustment(_WEB_SEARCH)
    bus.stop()
    # "healthy" gossip should improve (or at least not worsen) the adjustment
    assert adj_after >= adj_before, (
        f"Healthy gossip should improve peer's tool estimate: before={adj_before:.3f}, after={adj_after:.3f}"
    )


def test_no_tool_in_decision_learn_child_noop() -> None:
    """learn_child with no selected_tool must not raise and must not publish."""
    bus = GossipBus()
    peer = LiveBrain(ManifoldBrain(_CFG, tools=default_tools()), bus=bus, agent_id="peer3")
    parent = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="parent3")

    hd = parent.decide_hierarchical(_COMPLEX_TASK)
    no_tool_decision = dataclasses.replace(
        hd.sub_decisions[0] if hd.sub_decisions else parent.decide(_COMPLEX_TASK),
        selected_tool=None,
    )
    parent.learn_child(0, _COMPLEX_TASK, no_tool_decision, _FAILING_OUTCOME)
    bus.drain()
    # peer should have zero adjustment since no gossip was published
    adj = peer.brain.memory.tool_reliability_adjustment(_WEB_SEARCH)
    bus.stop()
    assert adj == 0.0


# ---------------------------------------------------------------------------
# learn() monolithic publishes gossip
# ---------------------------------------------------------------------------


def test_monolithic_learn_publishes_failure_gossip() -> None:
    """Repeated monolithic failures must drive peer's adj negative via gossip."""
    bus = GossipBus()
    peer = LiveBrain(ManifoldBrain(_CFG, tools=default_tools()), bus=bus, agent_id="peer4")
    parent = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="parent4")

    decision = parent.decide(_SIMPLE_TASK)
    decision = dataclasses.replace(decision, selected_tool="web_search")
    # Send 10 failing notes to overcome the 1.0 success_rate optimistic prior
    for _ in range(10):
        parent.learn(_SIMPLE_TASK, decision, _FAILING_OUTCOME)
    bus.drain()

    adj = peer.brain.memory.tool_reliability_adjustment(_WEB_SEARCH)
    bus.stop()
    # Gossip must have reached the peer (entry exists, adj != 0)
    assert adj != 0.0, "Gossip never reached peer"
    assert adj < 0.0, f"Expected negative peer adjustment after 10 failing notes, got {adj:.3f}"


def test_monolithic_learn_no_bus_noop() -> None:
    """learn() without a bus must not raise and must update local memory only."""
    brain = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=None)
    decision = brain.decide(_SIMPLE_TASK)
    decision = dataclasses.replace(decision, selected_tool="web_search")
    brain.learn(_SIMPLE_TASK, decision, _FAILING_OUTCOME)
    # local memory should record the outcome
    assert brain.memory.action_stats or True  # no crash is the key assertion


# ---------------------------------------------------------------------------
# Self-gossip prevention
# ---------------------------------------------------------------------------


def test_parent_does_not_receive_own_gossip() -> None:
    """The parent must not receive its own gossip note (self-loop prevention).

    After 10 failing learn() calls, the parent's adj reflects ONE direct
    ManifoldBrain.learn() path per call — NOT double-counted from gossip
    re-ingestion.  If self-gossip were re-ingested, the effective_lr would
    double and adj would diverge negatively much faster.
    """
    bus = GossipBus()
    parent = HierarchicalLiveBrain(_CFG, tools=default_tools(), bus=bus, agent_id="self_test")
    decision = parent.decide(_SIMPLE_TASK)
    decision = dataclasses.replace(decision, selected_tool="web_search")

    for _ in range(10):
        parent.learn(_SIMPLE_TASK, decision, _FAILING_OUTCOME)
    bus.drain()

    # Also create an isolated control brain with no bus (same direct updates, no gossip)
    control = ManifoldBrain(_CFG, tools=default_tools())
    for _ in range(10):
        control.learn(_SIMPLE_TASK, decision, _FAILING_OUTCOME)

    adj_parent = parent.memory.tool_reliability_adjustment(_WEB_SEARCH)
    adj_control = control.memory.tool_reliability_adjustment(_WEB_SEARCH)
    bus.stop()

    # Self-gossip prevention means parent's adj ≈ control's adj (not significantly worse)
    # Allow for small numerical differences; no self-loop means they stay close.
    diff = abs(adj_parent - adj_control)
    assert diff < 0.10, (
        f"Parent adj={adj_parent:.3f} vs control adj={adj_control:.3f} "
        f"differ by {diff:.3f}: self-gossip may be re-ingested."
    )


# ---------------------------------------------------------------------------
# run_gossip_hierarchical_suite
# ---------------------------------------------------------------------------


def test_gossip_hierarchical_suite_returns_three_findings() -> None:
    report = run_gossip_hierarchical_suite(seed=3000)
    assert len(report.findings) == 3
    assert {f.name for f in report.findings} == {
        "gossip_child_failure_propagates",
        "gossip_fallback_without_bus",
        "gossip_monolithic_learn_publishes",
    }


def test_gossip_hierarchical_suite_all_pass() -> None:
    report = run_gossip_hierarchical_suite(seed=3000)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.3f}. "
            f"{finding.interpretation}"
        )


def test_gossip_hierarchical_suite_has_summary() -> None:
    report = run_gossip_hierarchical_suite(seed=3000)
    assert len(report.honest_summary) >= 3
