from manifold import BrainConfig, BrainOutcome, BrainTask, ManifoldBrain, ToolProfile, decide_task


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
