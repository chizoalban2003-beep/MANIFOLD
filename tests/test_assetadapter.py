"""Tests for AssetAdapter — Phase 2.5: learning A from revealed user preferences."""

from manifold import (
    AssetAdapter,
    BrainConfig,
    BrainTask,
    LearnedPrices,
    ManifoldBrain,
    classify_user_signal,
    run_asset_learning_suite,
)


# ---------------------------------------------------------------------------
# classify_user_signal
# ---------------------------------------------------------------------------


def test_classify_correction_phrase() -> None:
    assert classify_user_signal("not what I asked") == "correction"


def test_classify_correction_wrong() -> None:
    assert classify_user_signal("that's wrong") == "correction"


def test_classify_correction_incorrect() -> None:
    assert classify_user_signal("incorrect answer") == "correction"


def test_classify_acceptance_thanks() -> None:
    assert classify_user_signal("thanks, that worked!") == "acceptance"


def test_classify_acceptance_perfect() -> None:
    assert classify_user_signal("perfect") == "acceptance"


def test_classify_acceptance_exactly() -> None:
    assert classify_user_signal("exactly what I needed") == "acceptance"


def test_classify_silence_no_followup() -> None:
    assert classify_user_signal(None, no_followup=True) == "silence"


def test_classify_ambiguous_none_no_timeout() -> None:
    assert classify_user_signal(None) == "ambiguous"


def test_classify_ambiguous_generic_text() -> None:
    assert classify_user_signal("hmm ok") == "ambiguous"


def test_classify_case_insensitive_correction() -> None:
    assert classify_user_signal("That Is Not what I wanted") == "correction"


def test_classify_no_followup_overrides_ambiguous_text() -> None:
    # Generic text + no_followup → silence takes precedence over ambiguous
    assert classify_user_signal("ok sure", no_followup=True) == "silence"


# ---------------------------------------------------------------------------
# AssetAdapter.observe_outcome
# ---------------------------------------------------------------------------


def test_adapter_correction_builds_negative_delta() -> None:
    adapter = AssetAdapter()
    for _ in range(20):
        adapter.observe_outcome("answer", "correction", stated_asset=0.70)
    delta = adapter.asset_corrections()["answer"].asset_delta
    assert delta < -0.10, f"Expected negative delta, got {delta:.3f}"


def test_adapter_acceptance_builds_positive_delta() -> None:
    adapter = AssetAdapter()
    for _ in range(20):
        adapter.observe_outcome("clarify", "acceptance", stated_asset=0.40)
    delta = adapter.asset_corrections()["clarify"].asset_delta
    assert delta > 0.05, f"Expected positive delta, got {delta:.3f}"


def test_adapter_silence_builds_small_positive_delta() -> None:
    adapter = AssetAdapter()
    # silence realized asset = 0.3 vs stated = 0.5 → gap = -0.2 → small negative or near zero
    for _ in range(20):
        adapter.observe_outcome("retrieve", "silence", stated_asset=0.30)
    delta = adapter.asset_corrections()["retrieve"].asset_delta
    # silence realized = 0.3, stated = 0.3 → gap = 0, delta stays near 0
    assert abs(delta) < 0.05, f"Silence with matching stated_asset should produce near-zero delta, got {delta:.3f}"


def test_adapter_ambiguous_produces_zero_observations() -> None:
    adapter = AssetAdapter()
    for _ in range(30):
        adapter.observe_outcome("verify", "ambiguous", stated_asset=0.60)
    corr = adapter.asset_corrections().get("verify")
    n = corr.n_observations if corr else 0
    assert n == 0, f"Ambiguous signals must not increment observations, got {n}"


def test_adapter_unknown_signal_also_skipped() -> None:
    adapter = AssetAdapter()
    adapter.observe_outcome("plan", "xyzzy", stated_asset=0.50)
    corr = adapter.asset_corrections().get("plan")
    n = corr.n_observations if corr else 0
    assert n == 0


def test_adapter_observation_count_increments_on_valid_signals() -> None:
    adapter = AssetAdapter()
    for sig in ("correction", "acceptance", "silence"):
        adapter.observe_outcome("answer", sig, stated_asset=0.5)
    count = adapter.asset_corrections()["answer"].n_observations
    assert count == 3


# ---------------------------------------------------------------------------
# AssetAdapter.adapt_asset
# ---------------------------------------------------------------------------


def test_adapt_asset_returns_stated_below_min_observations() -> None:
    adapter = AssetAdapter(min_observations=3)
    adapter.observe_outcome("answer", "correction", stated_asset=0.70)
    adapter.observe_outcome("answer", "correction", stated_asset=0.70)
    # only 2 observations, should still return stated
    assert adapter.adapt_asset("answer", 0.70) == 0.70


def test_adapt_asset_applies_correction_at_min_observations() -> None:
    adapter = AssetAdapter(min_observations=3)
    for _ in range(3):
        adapter.observe_outcome("answer", "correction", stated_asset=0.70)
    adapted = adapter.adapt_asset("answer", 0.70)
    assert adapted < 0.70, f"Expected adapted < stated after corrections, got {adapted:.3f}"


def test_adapt_asset_clamps_to_unit_interval_upper() -> None:
    adapter = AssetAdapter()
    for _ in range(30):
        adapter.observe_outcome("answer", "acceptance", stated_asset=0.95)
    adapted = adapter.adapt_asset("answer", 0.95)
    assert 0.0 <= adapted <= 1.0


def test_adapt_asset_clamps_to_unit_interval_lower() -> None:
    adapter = AssetAdapter()
    for _ in range(40):
        adapter.observe_outcome("answer", "correction", stated_asset=0.05)
    adapted = adapter.adapt_asset("answer", 0.05)
    assert 0.0 <= adapted <= 1.0


def test_adapt_asset_unknown_action_returns_stated() -> None:
    adapter = AssetAdapter()
    assert adapter.adapt_asset("stop", 0.50) == 0.50


# ---------------------------------------------------------------------------
# AssetAdapter.asset_corrections
# ---------------------------------------------------------------------------


def test_asset_corrections_returns_copy() -> None:
    adapter = AssetAdapter()
    adapter.observe_outcome("answer", "acceptance", stated_asset=0.5)
    copy1 = adapter.asset_corrections()
    copy1["injected"] = LearnedPrices()
    assert "injected" not in adapter.asset_corrections()


# ---------------------------------------------------------------------------
# ManifoldBrain.observe_asset + expected_action_utility integration
# ---------------------------------------------------------------------------

_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)


def test_brain_observe_asset_forwards_to_adapter() -> None:
    adapter = AssetAdapter()
    brain = ManifoldBrain(_CFG, asset_adapter=adapter)
    brain.observe_asset("answer", "correction", stated_asset=0.70)
    assert "answer" in adapter.asset_corrections()


def test_brain_observe_asset_noop_without_adapter() -> None:
    brain = ManifoldBrain(_CFG)  # no asset_adapter
    brain.observe_asset("answer", "correction", stated_asset=0.70)  # must not raise


def test_brain_expected_utility_decreases_after_corrections() -> None:
    adapter = AssetAdapter()
    brain = ManifoldBrain(_CFG, asset_adapter=adapter)
    task = BrainTask("Query", domain="general", stakes=0.70)
    # get baseline utility for "answer" before any learning
    baseline = brain.expected_action_utility(task, "answer", None, risk_score=0.20)
    # burn-in many corrections
    for _ in range(30):
        adapter.observe_outcome("answer", "correction", stated_asset=0.70)
    adapted = brain.expected_action_utility(task, "answer", None, risk_score=0.20)
    assert adapted < baseline, (
        f"Expected utility should decrease after corrections: {adapted:.3f} vs {baseline:.3f}"
    )


def test_brain_expected_utility_increases_after_acceptance() -> None:
    adapter = AssetAdapter()
    brain = ManifoldBrain(_CFG, asset_adapter=adapter)
    task = BrainTask("Query", domain="general", stakes=0.40)
    baseline = brain.expected_action_utility(task, "clarify", None, risk_score=0.30)
    for _ in range(30):
        adapter.observe_outcome("clarify", "acceptance", stated_asset=0.40)
    adapted = brain.expected_action_utility(task, "clarify", None, risk_score=0.30)
    assert adapted > baseline, (
        f"Expected utility should increase after acceptance: {adapted:.3f} vs {baseline:.3f}"
    )


# ---------------------------------------------------------------------------
# run_asset_learning_suite
# ---------------------------------------------------------------------------


def test_asset_learning_suite_returns_four_findings() -> None:
    report = run_asset_learning_suite(seed=2500)
    assert len(report.findings) == 4
    assert {f.name for f in report.findings} == {
        "asset_correction_learning",
        "asset_acceptance_learning",
        "asset_ambiguous_ignored",
        "classify_signal_accuracy",
    }


def test_asset_learning_suite_has_honest_summary() -> None:
    report = run_asset_learning_suite(seed=2500)
    assert len(report.honest_summary) >= 4


def test_asset_learning_suite_all_probes_pass() -> None:
    report = run_asset_learning_suite(seed=2500)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.3f}. "
            f"Interpretation: {finding.interpretation}"
        )
