"""Tests for PromptEncoder — Phase 4: raw prompt → BrainTask features."""

from manifold import (
    BrainConfig,
    BrainTask,
    ManifoldBrain,
    PromptEncoder,
    PromptFeatures,
    default_tools,
    run_encoder_suite,
)
from manifold.encoder import EncoderCorrection, _sum_to_signal, _length_complexity


# ---------------------------------------------------------------------------
# _sum_to_signal helper
# ---------------------------------------------------------------------------


def test_sum_to_signal_no_matches_returns_baseline() -> None:
    result = _sum_to_signal("zzzzxxx nothing here", [], baseline=0.5)
    assert result == 0.5


def test_sum_to_signal_positive_weights_raise_score() -> None:
    signals = [("hello", 0.5)]
    result = _sum_to_signal("hello world", signals, baseline=0.3)
    assert result > 0.3


def test_sum_to_signal_negative_weights_lower_score() -> None:
    signals = [("simple", -0.5)]
    result = _sum_to_signal("this is simple", signals, baseline=0.6)
    assert result < 0.6


def test_sum_to_signal_clamped_to_unit_interval() -> None:
    signals = [("x", 2.0), ("y", 2.0)]
    result = _sum_to_signal("x y z", signals, baseline=0.9)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _length_complexity helper
# ---------------------------------------------------------------------------


def test_length_complexity_very_short() -> None:
    assert _length_complexity("ok") < 0


def test_length_complexity_very_long() -> None:
    long_text = " ".join(["word"] * 100)
    assert _length_complexity(long_text) > 0


def test_length_complexity_medium() -> None:
    medium_text = " ".join(["word"] * 25)
    assert _length_complexity(medium_text) >= 0.0


# ---------------------------------------------------------------------------
# PromptEncoder.encode — keyword feature extraction
# ---------------------------------------------------------------------------

_ENC = PromptEncoder()


def test_encode_returns_prompt_features() -> None:
    f = _ENC.encode("hello world", "general")
    assert isinstance(f, PromptFeatures)
    assert f.prompt == "hello world"
    assert f.domain == "general"


def test_encode_all_features_in_unit_interval() -> None:
    f = _ENC.encode("This is a comprehensive deep-dive research analysis comparing multiple architectures", "research")
    for attr in ("complexity", "stakes", "uncertainty", "source_confidence", "tool_relevance"):
        val = getattr(f, attr)
        assert 0.0 <= val <= 1.0, f"{attr}={val:.3f} out of [0, 1]"


def test_encode_complexity_complex_vs_simple() -> None:
    enc = PromptEncoder()
    f_complex = enc.encode(
        "Provide a comprehensive step-by-step comparison of three cloud architectures for enterprise AI deployment",
        "general"
    )
    f_simple = enc.encode("What is 2 + 2?", "general")
    assert f_complex.complexity > f_simple.complexity


def test_encode_stakes_high_keywords() -> None:
    enc = PromptEncoder()
    f = enc.encode("Production outage affecting customer revenue — urgent!", "support")
    f_low = enc.encode("casual sandbox demo experiment", "support")
    assert f.stakes > f_low.stakes


def test_encode_uncertainty_ambiguous_keywords() -> None:
    enc = PromptEncoder()
    f_uncertain = enc.encode("Maybe this might work, I'm not sure — unclear requirements", "general")
    f_certain = enc.encode("Definitely always exactly confirmed proven standard specification", "general")
    assert f_uncertain.uncertainty > f_certain.uncertainty


def test_encode_source_confidence_official_vs_rumour() -> None:
    enc = PromptEncoder()
    f_official = enc.encode("According to the official RFC specification and verified database", "general")
    f_rumour = enc.encode("I heard some rumor from a blog that the old deprecated system", "general")
    assert f_official.source_confidence > f_rumour.source_confidence


def test_encode_tool_relevance_search_vs_opinion() -> None:
    enc = PromptEncoder()
    f_tool = enc.encode("Please search and fetch the latest API data from the database", "general")
    f_no_tool = enc.encode("What do you think about this? Your opinion please.", "general")
    assert f_tool.tool_relevance > f_no_tool.tool_relevance


def test_encode_raw_fields_present() -> None:
    enc = PromptEncoder()
    f = enc.encode("test prompt", "general")
    assert hasattr(f, "raw_complexity")
    assert hasattr(f, "raw_stakes")
    assert 0.0 <= f.raw_complexity <= 1.0
    assert 0.0 <= f.raw_stakes <= 1.0


def test_encode_domain_default_is_general() -> None:
    enc = PromptEncoder()
    f = enc.encode("some prompt")
    assert f.domain == "general"


# ---------------------------------------------------------------------------
# PromptFeatures.to_brain_task
# ---------------------------------------------------------------------------


def test_to_brain_task_basic() -> None:
    enc = PromptEncoder()
    features = enc.encode("Analyze security vulnerabilities", "security")
    task = features.to_brain_task()
    assert isinstance(task, BrainTask)
    assert task.domain == "security"
    assert task.complexity == features.complexity
    assert task.stakes == features.stakes


def test_to_brain_task_overrides() -> None:
    enc = PromptEncoder()
    features = enc.encode("some prompt", "coding")
    task = features.to_brain_task(time_pressure=0.9, safety_sensitivity=0.8)
    assert task.time_pressure == 0.9
    assert task.safety_sensitivity == 0.8
    # Base fields still come from encoder
    assert task.domain == "coding"


def test_to_brain_task_flows_to_manifold_brain() -> None:
    enc = PromptEncoder()
    features = enc.encode("Write a report on market trends", "research")
    task = features.to_brain_task(user_patience=0.7)
    brain = ManifoldBrain(BrainConfig(generations=2, population_size=12, grid_size=5), tools=default_tools())
    decision = brain.decide(task)
    assert decision.action in {
        "answer", "clarify", "retrieve", "verify", "use_tool",
        "delegate", "plan", "explore", "exploit", "wait", "escalate", "refuse", "stop",
    }


# ---------------------------------------------------------------------------
# PromptEncoder.update_from_price_delta
# ---------------------------------------------------------------------------


def test_price_delta_raises_complexity_correction() -> None:
    enc = PromptEncoder()
    for _ in range(15):
        enc.update_from_price_delta("legal", cost_delta=0.40)
    corr = enc.corrections().get("legal", {}).get("complexity")
    assert corr is not None
    assert corr.delta > 0.0
    assert corr.n_updates == 15


def test_price_delta_also_updates_uncertainty_for_risk() -> None:
    enc = PromptEncoder()
    for _ in range(15):
        enc.update_from_price_delta("finance", cost_delta=0.0, risk_delta=0.50)
    corr = enc.corrections().get("finance", {}).get("uncertainty")
    assert corr is not None
    assert corr.delta > 0.0


def test_price_delta_zero_produces_no_n_updates_for_uncertainty() -> None:
    enc = PromptEncoder()
    enc.update_from_price_delta("general", cost_delta=0.20, risk_delta=0.0)
    # With risk_delta=0.0, uncertainty should not be updated (the if-branch skips it)
    corr_u = enc.corrections().get("general", {}).get("uncertainty")
    assert corr_u is None


def test_correction_applies_to_subsequent_encode() -> None:
    enc = PromptEncoder()
    f_before = enc.encode("analyze legal contract", "legal")
    for _ in range(20):
        enc.update_from_price_delta("legal", cost_delta=0.50)
    f_after = enc.encode("analyze legal contract", "legal")
    assert f_after.complexity >= f_before.complexity, (
        f"Complexity should be ≥ before after positive corrections: {f_before.complexity:.3f} → {f_after.complexity:.3f}"
    )


# ---------------------------------------------------------------------------
# PromptEncoder.update_from_asset_delta
# ---------------------------------------------------------------------------


def test_negative_asset_delta_raises_stakes_correction() -> None:
    enc = PromptEncoder()
    for _ in range(15):
        enc.update_from_asset_delta("medical", "answer", asset_delta=-0.50)
    corr = enc.corrections().get("medical", {}).get("stakes")
    assert corr is not None
    assert corr.delta > 0.0


def test_positive_asset_delta_lowers_stakes_correction() -> None:
    enc = PromptEncoder()
    for _ in range(15):
        enc.update_from_asset_delta("demo", "answer", asset_delta=0.50)
    corr = enc.corrections().get("demo", {}).get("stakes")
    assert corr is not None
    assert corr.delta < 0.0


# ---------------------------------------------------------------------------
# PromptEncoder.corrections and reset_domain
# ---------------------------------------------------------------------------


def test_corrections_returns_copy() -> None:
    enc = PromptEncoder()
    enc.update_from_price_delta("test_domain", cost_delta=0.3)
    copy1 = enc.corrections()
    copy1["injected"] = {}
    assert "injected" not in enc.corrections()


def test_reset_domain_clears_corrections() -> None:
    enc = PromptEncoder()
    for _ in range(5):
        enc.update_from_price_delta("to_reset", cost_delta=0.4)
    assert enc.corrections().get("to_reset") is not None
    enc.reset_domain("to_reset")
    assert enc.corrections().get("to_reset") is None


def test_reset_domain_noop_for_unknown_domain() -> None:
    enc = PromptEncoder()
    enc.reset_domain("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# run_encoder_suite
# ---------------------------------------------------------------------------


def test_encoder_suite_returns_six_findings() -> None:
    report = run_encoder_suite(seed=4000)
    assert len(report.findings) == 6
    assert {f.name for f in report.findings} == {
        "encoder_complexity_separation",
        "encoder_stakes_separation",
        "encoder_complexity_ema_from_price",
        "encoder_stakes_ema_from_asset",
        "encoder_to_brain_task",
        "encoder_reset_domain",
    }


def test_encoder_suite_has_honest_summary() -> None:
    report = run_encoder_suite(seed=4000)
    assert len(report.honest_summary) >= 5


def test_encoder_suite_all_probes_pass() -> None:
    report = run_encoder_suite(seed=4000)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.4f}. "
            f"{finding.interpretation}"
        )
