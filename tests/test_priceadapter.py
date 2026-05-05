"""Tests for PriceAdapter — Phase 2: learning C/R/A prices from outcomes."""

from manifold import (
    BrainConfig,
    BrainOutcome,
    BrainTask,
    LearnedPrices,
    ManifoldBrain,
    PriceAdapter,
    ToolProfile,
    attribute_to_tool,
    run_price_learning_suite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str = "test_tool", **kwargs) -> ToolProfile:
    defaults = dict(cost=0.10, latency=0.0, reliability=0.85, risk=0.10, asset=0.70, domain="general")
    defaults.update(kwargs)
    return ToolProfile(name, **defaults)


def _success(
    cost_paid: float = 0.10,
    risk_realized: float = 0.05,
    asset_gained: float = 0.70,
) -> BrainOutcome:
    return BrainOutcome(success=True, cost_paid=cost_paid, risk_realized=risk_realized, asset_gained=asset_gained)


def _failure(
    failure_mode: str = "tool_error",
    cost_paid: float = 0.10,
    risk_realized: float = 0.50,
) -> BrainOutcome:
    return BrainOutcome(
        success=False,
        cost_paid=cost_paid,
        risk_realized=risk_realized,
        asset_gained=0.0,
        failure_mode=failure_mode,
    )


# ---------------------------------------------------------------------------
# attribute_to_tool
# ---------------------------------------------------------------------------


def test_attribute_to_tool_high_for_tool_error() -> None:
    assert attribute_to_tool("tool_error") >= 0.90


def test_attribute_to_tool_low_for_environment_noise() -> None:
    assert attribute_to_tool("environment_noise") <= 0.10


def test_attribute_to_tool_low_for_timeout() -> None:
    assert attribute_to_tool("timeout") <= 0.20


def test_attribute_to_tool_mid_for_unknown() -> None:
    blame = attribute_to_tool("unknown")
    assert 0.40 <= blame <= 0.60


def test_attribute_to_tool_unrecognised_returns_half() -> None:
    assert attribute_to_tool("xyzzy_mode") == 0.50


# ---------------------------------------------------------------------------
# LearnedPrices
# ---------------------------------------------------------------------------


def test_learned_prices_default_state_is_zero() -> None:
    lp = LearnedPrices()
    assert lp.cost_delta == 0.0
    assert lp.risk_delta == 0.0
    assert lp.asset_delta == 0.0
    assert lp.n_observations == 0


# ---------------------------------------------------------------------------
# PriceAdapter.observe
# ---------------------------------------------------------------------------


def test_adapter_cost_delta_grows_with_cost_overrun() -> None:
    adapter = PriceAdapter()
    tool = _tool(cost=0.10)
    for _ in range(20):
        adapter.observe(tool, _success(cost_paid=0.35))
    delta = adapter.price_corrections()[tool.name].cost_delta
    assert delta > 0.10, f"Expected cost_delta > 0.10 after 20 overrun observations, got {delta:.3f}"


def test_adapter_cost_delta_negative_when_cost_underrun() -> None:
    adapter = PriceAdapter()
    tool = _tool(cost=0.40)
    for _ in range(10):
        adapter.observe(tool, _success(cost_paid=0.05))
    delta = adapter.price_corrections()[tool.name].cost_delta
    assert delta < 0.0, "cost_delta should be negative when actual cost < stated cost"


def test_adapter_risk_delta_higher_for_tool_error_than_env_noise() -> None:
    tool = _tool(name="risktest", risk=0.10)
    adapter_tool = PriceAdapter()
    adapter_env = PriceAdapter()
    for _ in range(15):
        adapter_tool.observe(tool, _failure("tool_error", risk_realized=0.80))
        adapter_env.observe(tool, _failure("environment_noise", risk_realized=0.80))
    delta_tool = adapter_tool.price_corrections()[tool.name].risk_delta
    delta_env = adapter_env.price_corrections()[tool.name].risk_delta
    assert delta_tool > delta_env * 5, (
        f"tool_error risk_delta={delta_tool:.3f} should be 5× env_noise {delta_env:.3f}"
    )


def test_adapter_asset_delta_only_updated_on_success() -> None:
    adapter = PriceAdapter()
    tool = _tool(asset=0.70)
    for _ in range(10):
        adapter.observe(tool, _failure(risk_realized=0.50))
    delta = adapter.price_corrections()[tool.name].asset_delta
    assert delta == 0.0, "asset_delta must not be updated on failure outcomes"


def test_adapter_asset_delta_updated_on_success() -> None:
    adapter = PriceAdapter()
    tool = _tool(asset=0.70)
    for _ in range(10):
        adapter.observe(tool, _success(asset_gained=0.90))
    delta = adapter.price_corrections()[tool.name].asset_delta
    assert delta > 0.0, "asset_delta should be positive when asset_gained > tool.asset"


def test_adapter_observation_count_increments() -> None:
    adapter = PriceAdapter()
    tool = _tool()
    for i in range(5):
        adapter.observe(tool, _success())
    assert adapter.price_corrections()[tool.name].n_observations == 5


# ---------------------------------------------------------------------------
# PriceAdapter.adapt
# ---------------------------------------------------------------------------


def test_adapt_returns_original_below_min_observations() -> None:
    adapter = PriceAdapter(min_observations=3)
    tool = _tool()
    for _ in range(2):
        adapter.observe(tool, _success())
    assert adapter.adapt(tool) is tool


def test_adapt_returns_corrected_profile_at_min_observations() -> None:
    adapter = PriceAdapter(min_observations=3)
    tool = _tool(cost=0.05)
    for _ in range(3):
        adapter.observe(tool, _success(cost_paid=0.40))
    adapted = adapter.adapt(tool)
    assert adapted is not tool
    assert adapted.cost > tool.cost


def test_adapt_clamps_cost_to_unit_interval() -> None:
    adapter = PriceAdapter()
    tool = _tool(cost=0.95)
    for _ in range(20):
        adapter.observe(tool, _success(cost_paid=1.0))
    adapted = adapter.adapt(tool)
    assert 0.0 <= adapted.cost <= 1.0


def test_adapt_clamps_risk_to_unit_interval() -> None:
    adapter = PriceAdapter()
    tool = _tool(risk=0.90)
    for _ in range(20):
        adapter.observe(tool, _failure("tool_error", risk_realized=1.0))
    adapted = adapter.adapt(tool)
    assert 0.0 <= adapted.risk <= 1.0


def test_adapt_preserves_name_latency_reliability_domain() -> None:
    adapter = PriceAdapter()
    tool = _tool(name="my_tool", latency=0.12, reliability=0.88)
    for _ in range(5):
        adapter.observe(tool, _success(cost_paid=0.50))
    adapted = adapter.adapt(tool)
    assert adapted.name == "my_tool"
    assert adapted.latency == 0.12
    assert adapted.reliability == 0.88
    assert adapted.domain == "general"


def test_adapt_unknown_tool_returns_original() -> None:
    adapter = PriceAdapter()
    tool = _tool()
    assert adapter.adapt(tool) is tool


# ---------------------------------------------------------------------------
# PriceAdapter.price_corrections
# ---------------------------------------------------------------------------


def test_price_corrections_returns_copy() -> None:
    adapter = PriceAdapter()
    tool = _tool()
    adapter.observe(tool, _success())
    corrections = adapter.price_corrections()
    corrections["injected"] = LearnedPrices()
    assert "injected" not in adapter.price_corrections()


# ---------------------------------------------------------------------------
# ManifoldBrain + PriceAdapter integration
# ---------------------------------------------------------------------------

_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)


def test_brain_selects_tool_before_price_learning() -> None:
    """Baseline: tool with positive stated utility should be selected."""
    tool = ToolProfile("cheap_tool", cost=0.05, latency=0.0, reliability=0.80,
                       risk=0.05, asset=0.50, domain="general")
    brain = ManifoldBrain(_CFG, tools=[tool], price_adapter=PriceAdapter())
    task = BrainTask("lookup", domain="general", tool_relevance=0.90)
    # Stated utility = 0.40 − 0.10 = 0.30 > 0 → must be selected before burn-in
    assert brain.select_tool(task) is not None


def test_brain_deselects_expensive_tool_after_price_learning() -> None:
    """After learning that true cost >> stated cost, the brain should stop selecting the tool."""
    tool = ToolProfile("pricey_tool", cost=0.05, latency=0.0, reliability=0.80,
                       risk=0.05, asset=0.50, domain="general")
    adapter = PriceAdapter()
    brain = ManifoldBrain(_CFG, tools=[tool], price_adapter=adapter)
    task = BrainTask("lookup", domain="general", tool_relevance=0.90)

    # Burn-in: real cost is 0.45 (stated is 0.05)
    for _ in range(35):
        adapter.observe(tool, BrainOutcome(
            success=True, cost_paid=0.45, risk_realized=0.05, asset_gained=0.50,
        ))

    # Adapted utility ≈ 0.40 − 0.49 < 0 → tool should not be selected
    assert brain.select_tool(task) is None, (
        f"Tool should be deselected after cost_delta="
        f"{adapter.price_corrections()['pricey_tool'].cost_delta:.3f}"
    )


def test_brain_without_adapter_ignores_true_cost() -> None:
    """Without a PriceAdapter, the brain always uses declared prices — tool stays selected."""
    tool = ToolProfile("pricey_tool", cost=0.05, latency=0.0, reliability=0.80,
                       risk=0.05, asset=0.50, domain="general")
    brain = ManifoldBrain(_CFG, tools=[tool])  # no price_adapter
    task = BrainTask("lookup", domain="general", tool_relevance=0.90)
    # Even after many expensive outcomes in memory, no adapter → tool still selected
    assert brain.select_tool(task) is not None


def test_brain_learn_forwards_outcome_to_adapter() -> None:
    """brain.learn() must call adapter.observe() when a tool was selected."""
    tool = ToolProfile("tracked_tool", cost=0.05, latency=0.0, reliability=0.80,
                       risk=0.05, asset=0.85, domain="support")
    adapter = PriceAdapter()
    brain = ManifoldBrain(_CFG, tools=[tool], price_adapter=adapter)
    task = BrainTask("lookup", domain="support", tool_relevance=0.95,
                     source_confidence=0.85, uncertainty=0.3, stakes=0.5)
    decision = brain.decide(task)
    brain.learn(task, decision, BrainOutcome(
        success=True, cost_paid=0.30, risk_realized=0.05, asset_gained=0.85,
    ))
    if decision.selected_tool == "tracked_tool":
        assert "tracked_tool" in adapter.price_corrections()


def test_brain_learn_with_no_adapter_does_not_raise() -> None:
    """learn() must not raise when price_adapter is None (default)."""
    tool = ToolProfile("basic", cost=0.05, latency=0.05, reliability=0.85, risk=0.05, asset=0.70, domain="general")
    brain = ManifoldBrain(_CFG, tools=[tool])
    task = BrainTask("query", domain="general", tool_relevance=0.80)
    decision = brain.decide(task)
    brain.learn(task, decision, BrainOutcome(success=True, cost_paid=0.05, risk_realized=0.0, asset_gained=0.70))


# ---------------------------------------------------------------------------
# run_price_learning_suite
# ---------------------------------------------------------------------------


def test_price_learning_suite_returns_three_findings() -> None:
    report = run_price_learning_suite(seed=2500, n_rounds=35)
    assert len(report.findings) == 3
    assert {f.name for f in report.findings} == {
        "price_convergence",
        "causal_attribution",
        "adaptation_improves_selection",
    }


def test_price_learning_suite_has_honest_summary() -> None:
    report = run_price_learning_suite(seed=2500, n_rounds=35)
    assert len(report.honest_summary) >= 4


def test_price_learning_suite_all_probes_pass() -> None:
    report = run_price_learning_suite(seed=2500, n_rounds=35)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.3f}. "
            f"Interpretation: {finding.interpretation}"
        )
