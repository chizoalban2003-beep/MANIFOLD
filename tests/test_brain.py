from manifold import BrainConfig, BrainMemory, BrainOutcome, BrainTask, ManifoldBrain, ToolProfile, decide_task


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
