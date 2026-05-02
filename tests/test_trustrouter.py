from manifold import DialogueTask, TrustRouter, TrustRouterConfig, route_task


def test_trustrouter_maps_task_to_world() -> None:
    router = TrustRouter(TrustRouterConfig(generations=2, population_size=12, grid_size=5))
    task = DialogueTask(
        prompt="Should I answer directly?",
        domain="support",
        uncertainty=0.7,
        complexity=0.4,
        stakes=0.8,
        source_confidence=0.5,
        dynamic_intent=True,
    )

    world = router.map_task_to_world(task)

    assert world.targets[0].id == "resolved_intent"
    assert world.targets[0].moves == "random_walk"
    assert {rule.triggers for rule in world.rules} == {"trusted_lie", "miss_target", "low_energy"}


def test_trustrouter_routes_low_and_high_risk_differently() -> None:
    router = TrustRouter(TrustRouterConfig(generations=3, population_size=16, grid_size=5))
    low = DialogueTask(
        prompt="Say hello",
        domain="chat",
        uncertainty=0.1,
        complexity=0.1,
        stakes=0.1,
        source_confidence=0.95,
        user_patience=0.8,
    )
    high = DialogueTask(
        prompt="Give legal medical advice with uncertain sources",
        domain="regulated",
        uncertainty=0.9,
        complexity=0.8,
        stakes=0.95,
        source_confidence=0.2,
        user_patience=0.6,
        safety_sensitivity=0.9,
        dynamic_intent=True,
    )

    low_decision = router.route(low)
    high_decision = router.route(high)

    assert low_decision.risk_score < high_decision.risk_score
    assert low_decision.action in {"answer", "verify", "retrieve"}
    assert high_decision.action in {"clarify", "retrieve", "verify", "escalate", "refuse"}


def test_trustrouter_updates_learning_memory() -> None:
    router = TrustRouter(TrustRouterConfig(generations=2, population_size=12, grid_size=5))
    task = DialogueTask(
        prompt="Ambiguous billing issue",
        domain="support",
        uncertainty=0.8,
        complexity=0.4,
        stakes=0.6,
        source_confidence=0.4,
    )

    first = router.route(task)
    second = router.route(task)

    assert router.memory.domain_stats["support"]["count"] == 2.0
    assert second.risk_score >= first.risk_score - 0.2


def test_route_task_convenience_api() -> None:
    decision = route_task(
        DialogueTask(
            prompt="Need a sourced answer",
            uncertainty=0.6,
            complexity=0.5,
            stakes=0.5,
            source_confidence=0.3,
        ),
        TrustRouterConfig(generations=2, population_size=12, grid_size=5),
    )

    assert 0.0 <= decision.confidence <= 1.0
    assert decision.result.history
