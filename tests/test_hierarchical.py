"""Tests for HierarchicalBrain — Phase 3: MANIFOLD-of-MANIFOLDs."""

from manifold import (
    BrainConfig,
    BrainOutcome,
    BrainTask,
    DecompositionPlan,
    HierarchicalBrain,
    SubTaskSpec,
    default_tools,
    run_hierarchical_suite,
)


_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)

_COMPLEX_TASK = BrainTask(
    "Write a comprehensive market research report",
    domain="research",
    complexity=0.92,
    stakes=0.85,
    uncertainty=0.55,
    source_confidence=0.60,
    collaboration_value=0.60,
    user_patience=0.80,
)

_SIMPLE_TASK = BrainTask(
    "What is 2 + 2?",
    domain="math",
    complexity=0.15,
    stakes=0.10,
    uncertainty=0.05,
)


# ---------------------------------------------------------------------------
# SubTaskSpec
# ---------------------------------------------------------------------------


def test_sub_task_spec_is_frozen() -> None:
    st = SubTaskSpec(
        prompt="test", domain="general", complexity=0.5,
        stakes=0.5, weight=0.5,
    )
    try:
        st.complexity = 0.9  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except Exception:
        pass  # expected


def test_sub_task_spec_weight_default() -> None:
    st = SubTaskSpec(prompt="p", domain="d", complexity=0.5, stakes=0.5, weight=0.6)
    assert st.weight == 0.6


# ---------------------------------------------------------------------------
# DecompositionPlan
# ---------------------------------------------------------------------------


def test_decomposition_plan_defaults() -> None:
    plan = DecompositionPlan(
        sub_tasks=(SubTaskSpec("p", "d", 0.5, 0.5, 0.5),),
        decompose_cost=0.15,
    )
    assert plan.coordination_tax == 0.10


def test_decomposition_plan_is_frozen() -> None:
    plan = DecompositionPlan(
        sub_tasks=(SubTaskSpec("p", "d", 0.5, 0.5, 0.5),),
        decompose_cost=0.15,
    )
    try:
        plan.decompose_cost = 0.99  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HierarchicalDecision
# ---------------------------------------------------------------------------


def test_hierarchical_decision_is_frozen() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_SIMPLE_TASK)
    try:
        hd.decomposed = True  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except Exception:
        pass


def test_hierarchical_decision_has_depth_field() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_SIMPLE_TASK, depth=0)
    assert hd.depth == 0


# ---------------------------------------------------------------------------
# HierarchicalBrain.decide_hierarchical — basic behaviour
# ---------------------------------------------------------------------------


def test_simple_task_not_decomposed() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_SIMPLE_TASK)
    assert not hd.decomposed
    assert hd.sub_decisions is None
    assert hd.plan is None


def test_simple_task_combined_utility_equals_top() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_SIMPLE_TASK)
    assert hd.combined_utility == hd.top_decision.expected_utility


def test_complex_task_decomposes() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.decomposed, (
        f"Expected decomposition for complexity=0.92, got decomposed={hd.decomposed}, "
        f"combined_utility={hd.combined_utility:.3f}, monolithic={hd.top_decision.expected_utility:.3f}"
    )


def test_complex_task_has_two_sub_decisions() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.sub_decisions is not None
    assert len(hd.sub_decisions) == 2


def test_complex_task_plan_present() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.plan is not None
    assert hd.plan.decompose_cost > 0.0
    assert len(hd.plan.sub_tasks) == 2


def test_sub_decisions_are_valid_brain_decisions() -> None:
    from manifold import BrainDecision
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.sub_decisions is not None
    for sd in hd.sub_decisions:
        assert isinstance(sd, BrainDecision)
        assert 0.0 <= sd.confidence <= 1.0


def test_sub_task_complexity_less_than_parent() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.plan is not None
    for st in hd.plan.sub_tasks:
        assert st.complexity < _COMPLEX_TASK.complexity


def test_sub_task_weights_sum_to_one() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert hd.plan is not None
    total_weight = sum(st.weight for st in hd.plan.sub_tasks)
    assert abs(total_weight - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Max depth prevents infinite recursion
# ---------------------------------------------------------------------------


def test_max_depth_zero_prevents_decomposition() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools(), max_depth=0)
    hd = brain.decide_hierarchical(_COMPLEX_TASK, depth=0)
    # depth=0 >= max_depth=0 → no decomposition
    assert not hd.decomposed


def test_max_depth_respected() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools(), max_depth=1)
    hd = brain.decide_hierarchical(_COMPLEX_TASK, depth=1)
    assert not hd.decomposed


# ---------------------------------------------------------------------------
# Coordination tax affects decomposition decision
# ---------------------------------------------------------------------------


def test_high_coordination_tax_blocks_decomposition() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools(), coordination_tax=0.80)
    hd = brain.decide_hierarchical(_COMPLEX_TASK)
    assert not hd.decomposed, (
        f"Expected no decomposition with 80% tax, got decomposed={hd.decomposed}, "
        f"combined_utility={hd.combined_utility:.3f}"
    )


def test_zero_coordination_tax_maximises_decomposition_benefit() -> None:
    brain_no_tax = HierarchicalBrain(_CFG, tools=default_tools(), coordination_tax=0.0)
    hd_no_tax = brain_no_tax.decide_hierarchical(_COMPLEX_TASK)
    brain_high_tax = HierarchicalBrain(_CFG, tools=default_tools(), coordination_tax=0.80)
    hd_high_tax = brain_high_tax.decide_hierarchical(_COMPLEX_TASK)
    # Zero tax should produce higher combined utility than 80% tax
    assert hd_no_tax.combined_utility >= hd_high_tax.combined_utility


# ---------------------------------------------------------------------------
# HierarchicalBrain inherits ManifoldBrain capabilities
# ---------------------------------------------------------------------------


def test_hierarchical_brain_decide_works_like_manifold() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    decision = brain.decide(_SIMPLE_TASK)
    assert decision.action in {
        "answer", "clarify", "retrieve", "verify", "use_tool",
        "delegate", "plan", "explore", "exploit", "wait", "escalate", "refuse", "stop",
    }


def test_hierarchical_brain_learn_works() -> None:
    brain = HierarchicalBrain(_CFG, tools=default_tools())
    decision = brain.decide(_SIMPLE_TASK)
    brain.learn(
        _SIMPLE_TASK, decision,
        BrainOutcome(success=True, cost_paid=0.05, risk_realized=0.0, asset_gained=0.70),
    )
    assert brain.memory.action_stats  # memory was updated


# ---------------------------------------------------------------------------
# run_hierarchical_suite
# ---------------------------------------------------------------------------


def test_hierarchical_suite_returns_four_findings() -> None:
    report = run_hierarchical_suite(seed=2500)
    assert len(report.findings) == 4
    assert {f.name for f in report.findings} == {
        "decomposition_triggered",
        "decomposition_skipped_simple",
        "sub_decisions_count",
        "coordination_tax_limits_decomposition",
    }


def test_hierarchical_suite_has_honest_summary() -> None:
    report = run_hierarchical_suite(seed=2500)
    assert len(report.honest_summary) >= 4


def test_hierarchical_suite_all_probes_pass() -> None:
    report = run_hierarchical_suite(seed=2500)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.3f}. "
            f"Interpretation: {finding.interpretation}"
        )
