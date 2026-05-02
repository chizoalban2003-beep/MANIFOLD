from manifold import BrainConfig, BrainMemory, BrainOutcome, BrainTask, GossipNote, ManifoldBrain, ScoutRecord, ToolProfile, decide_task


def test_brain_selects_useful_tool() -> None:
    brain = ManifoldBrain(
        BrainConfig(generations=2, population_size=12, grid_size=5),
        tools=[
            ToolProfile(
                "calculator",
                cost=0.04,
                latency=0.04,
                reliability=0.96,
                risk=0.02,
                asset=0.8,
                domain="math",
            )
        ],
    )
    task = BrainTask(
        "Calculate this",
        domain="math",
        uncertainty=0.3,
        complexity=0.5,
        stakes=0.5,
        source_confidence=0.8,
        tool_relevance=0.95,
    )

    decision = brain.decide(task)

    assert decision.action == "use_tool"
    assert decision.selected_tool == "calculator"


def test_brain_refuses_high_safety_task() -> None:
    decision = decide_task(
        BrainTask(
            "Unsafe high stakes request",
            domain="safety",
            uncertainty=0.9,
            complexity=0.7,
            stakes=0.9,
            source_confidence=0.3,
            safety_sensitivity=0.95,
            dynamic_goal=True,
        ),
        BrainConfig(generations=2, population_size=12, grid_size=5),
    )

    assert decision.action == "refuse"


def test_brain_memory_updates_domain_action_and_tool() -> None:
    brain = ManifoldBrain(
        BrainConfig(generations=2, population_size=12, grid_size=5),
        tools=[
            ToolProfile(
                "retriever",
                cost=0.1,
                latency=0.1,
                reliability=0.9,
                risk=0.05,
                asset=0.8,
                domain="research",
            )
        ],
    )
    task = BrainTask("Find sources", domain="research", tool_relevance=0.9, source_confidence=0.3)
    decision = brain.decide(task)

    brain.learn(task, decision, BrainOutcome(True, cost_paid=0.2, risk_realized=0.1, asset_gained=1.0))

    assert brain.memory.domain_stats["research"]["count"] == 1.0
    assert brain.memory.action_stats[decision.action]["count"] == 1.0
    if decision.selected_tool:
        assert brain.memory.tool_stats[decision.selected_tool]["count"] == 1.0


def test_tool_scars_depend_on_failure_mode_and_decay() -> None:
    memory = BrainMemory()
    brain = ManifoldBrain(
        BrainConfig(generations=2, population_size=12, grid_size=5),
        tools=[
            ToolProfile(
                "order_lookup",
                cost=0.05,
                latency=0.05,
                reliability=0.9,
                risk=0.04,
                asset=0.85,
                domain="support",
            )
        ],
        memory=memory,
    )
    task = BrainTask("Check order", domain="support", tool_relevance=0.95, source_confidence=0.8)
    decision = brain.decide(task)

    brain.learn(
        task,
        decision,
        BrainOutcome(False, cost_paid=0.2, risk_realized=0.8, asset_gained=0.0, failure_mode="tool_error"),
    )
    deep_scar = memory.tool_stats["order_lookup"]["success_rate"]
    memory.decay(rate=0.5)
    recovered = memory.tool_stats["order_lookup"]["success_rate"]

    assert deep_scar < 0.9
    assert recovered > deep_scar


def test_tool_can_recover_after_outage_decay() -> None:
    brain = ManifoldBrain(
        BrainConfig(generations=2, population_size=12, grid_size=5),
        tools=[
            ToolProfile(
                "order_lookup",
                cost=0.05,
                latency=0.05,
                reliability=0.9,
                risk=0.04,
                asset=0.85,
                domain="support",
            )
        ],
    )
    task = BrainTask(
        "Check order",
        domain="support",
        uncertainty=0.35,
        complexity=0.35,
        stakes=0.55,
        source_confidence=0.75,
        tool_relevance=0.95,
        time_pressure=0.6,
        safety_sensitivity=0.05,
        user_patience=0.8,
    )
    before = brain.decide(task)
    for _ in range(7):
        brain.learn(
            task,
            before,
            BrainOutcome(
                success=False,
                cost_paid=0.2,
                risk_realized=0.75,
                asset_gained=0.0,
                rule_violations=1,
                failure_mode="tool_error",
            ),
        )

    during_outage = brain.decide(task)
    for _ in range(10):
        brain.memory.decay(0.15)
    after_recovery = brain.decide(task)

    assert before.action == "use_tool"
    assert during_outage.action != "use_tool"
    assert after_recovery.action == "use_tool"


# ---------------------------------------------------------------------------
# Gossip-weighting tests
# ---------------------------------------------------------------------------


def test_single_gossip_cannot_kill_a_tool() -> None:
    """One malicious gossip note must not cause the Brain to abandon a healthy tool."""
    brain = ManifoldBrain(
        BrainConfig(generations=2, population_size=12, grid_size=5),
        tools=[
            ToolProfile(
                "database",
                cost=0.04,
                latency=0.05,
                reliability=0.95,
                risk=0.03,
                asset=0.9,
                domain="support",
            )
        ],
    )
    task = BrainTask(
        "Look up record",
        domain="support",
        tool_relevance=0.95,
        source_confidence=0.85,
    )

    # Confirm baseline: tool is used before any gossip
    before = brain.decide(task)
    assert before.action == "use_tool"

    # Single predatory scout claims the tool is failing
    brain.memory.ingest_gossip(
        GossipNote(
            tool="database",
            claim="failing",
            source_id="scout_99",
            source_reputation=0.81,
            source_is_scout=True,
            confidence=0.73,
            age_minutes=5.0,
        )
    )

    # Tool reputation should barely move; Brain still chooses use_tool
    after_one_note = brain.decide(task)
    assert after_one_note.action == "use_tool"


def test_consensus_gossip_has_meaningful_impact() -> None:
    """Three independent non-scout notes with sufficient weight must lower tool success_rate."""
    memory = BrainMemory()
    tool_name = "order_lookup"
    initial_rate = memory.tool_stats.get(tool_name, {}).get("success_rate", 1.0)

    for i in range(3):
        memory.ingest_gossip(
            GossipNote(
                tool=tool_name,
                claim="failing",
                source_id=f"agent_{i}",
                source_reputation=0.85,
                source_is_scout=False,
                confidence=0.90,
                age_minutes=1.0,
            )
        )

    final_rate = memory.tool_stats[tool_name]["success_rate"]
    assert final_rate < initial_rate, "Consensus gossip should reduce tool success_rate"


def test_scout_discount_reduces_gossip_impact_vs_non_scout() -> None:
    """Scout gossip should have less impact than equivalent non-scout gossip."""
    note_base = dict(
        tool="web_search",
        claim="failing",
        source_id="source_A",
        source_reputation=0.81,
        confidence=1.0,
        age_minutes=0.0,
    )

    mem_scout = BrainMemory()
    mem_scout.ingest_gossip(GossipNote(**note_base, source_is_scout=True))
    rate_scout = mem_scout.tool_stats["web_search"]["success_rate"]

    mem_non_scout = BrainMemory()
    mem_non_scout.ingest_gossip(GossipNote(**note_base, source_is_scout=False))
    rate_non_scout = mem_non_scout.tool_stats["web_search"]["success_rate"]

    # Scout gossip should leave a smaller scar (higher remaining success_rate)
    assert rate_scout > rate_non_scout


def test_scout_promotion_lifts_discount() -> None:
    """A scout that is right > 80% of the time over 50+ predictions earns a higher discount."""
    record = ScoutRecord()
    assert record.discount == 0.7  # default

    # Log 50 correct predictions
    for _ in range(50):
        record.log_prediction(True)

    assert record.precision > 0.80
    assert record.discount == 0.9  # promoted


def test_scout_without_enough_claims_stays_at_default_discount() -> None:
    record = ScoutRecord()
    for _ in range(49):
        record.log_prediction(True)

    assert record.discount == 0.7  # not yet promoted


def test_gossip_age_decay_reduces_weight() -> None:
    """Older gossip notes should have less impact than fresh ones."""
    tool_name = "retriever"

    mem_fresh = BrainMemory()
    mem_fresh.ingest_gossip(
        GossipNote(
            tool=tool_name,
            claim="failing",
            source_id="agent_1",
            source_reputation=0.90,
            source_is_scout=False,
            age_minutes=0.0,
        )
    )
    rate_fresh = mem_fresh.tool_stats[tool_name]["success_rate"]

    mem_old = BrainMemory()
    mem_old.ingest_gossip(
        GossipNote(
            tool=tool_name,
            claim="failing",
            source_id="agent_1",
            source_reputation=0.90,
            source_is_scout=False,
            age_minutes=60.0,  # 1 hour old
        )
    )
    rate_old = mem_old.tool_stats[tool_name]["success_rate"]

    # Older note should leave a smaller scar (higher remaining success_rate)
    assert rate_old > rate_fresh


def test_scout_accuracy_is_tracked_when_actual_outcome_provided() -> None:
    """Providing actual_outcome to ingest_gossip logs the scout's prediction accuracy."""
    memory = BrainMemory()
    note = GossipNote(
        tool="code_runner",
        claim="failing",
        source_id="scout_42",
        source_reputation=0.75,
        source_is_scout=True,
        age_minutes=2.0,
    )

    # Gossip says "failing" and the tool actually did fail (actual_outcome=False)
    memory.ingest_gossip(note, actual_outcome=False)

    record = memory.scout_tracker["scout_42"]
    assert len(record._predictions) == 1
    assert record._predictions[0] is True  # scout was correct


def test_gossip_healthy_claim_raises_success_rate() -> None:
    """A 'healthy' gossip note after scars should nudge success_rate upward."""
    memory = BrainMemory()
    tool_name = "calculator"

    # First, apply a real failure scar directly
    memory.tool_stats[tool_name] = {
        "count": 5.0,
        "success_rate": 0.50,
        "utility": -0.3,
        "consecutive_failures": 3.0,
    }

    memory.ingest_gossip(
        GossipNote(
            tool=tool_name,
            claim="healthy",
            source_id="agent_5",
            source_reputation=0.88,
            source_is_scout=False,
            age_minutes=0.0,
        )
    )

    assert memory.tool_stats[tool_name]["success_rate"] > 0.50
