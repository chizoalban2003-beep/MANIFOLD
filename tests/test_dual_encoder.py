"""Tests for Phase 5: DualPathEncoder, SemanticBridge, PromptCluster, and encoder extensions."""

import pytest

from manifold import (
    DualPathEncoder,
    PromptCluster,
    PromptEncoder,
    PromptFeatures,
    SemanticBridge,
    run_dual_encoder_suite,
)
from manifold.encoder import (
    _bow_vector,
    _cosine_similarity,
    _length_complexity,
    _sum_to_signal,
    _tokenize,
    _DEFAULT_CLUSTERS,
)


# ---------------------------------------------------------------------------
# _tokenize helper
# ---------------------------------------------------------------------------


def test_tokenize_lowercases() -> None:
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation() -> None:
    assert "gdpr" in _tokenize("GDPR, compliance!")


def test_tokenize_empty() -> None:
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _bow_vector
# ---------------------------------------------------------------------------


def test_bow_vector_known_word() -> None:
    vocab = {"hello": 0, "world": 1}
    vec = _bow_vector(["hello", "hello", "world"], vocab)
    assert vec[0] == 2.0
    assert vec[1] == 1.0


def test_bow_vector_unknown_word() -> None:
    vocab = {"hello": 0}
    vec = _bow_vector(["unknown"], vocab)
    assert vec[0] == 0.0


def test_bow_vector_empty_tokens() -> None:
    vocab = {"hello": 0}
    vec = _bow_vector([], vocab)
    assert vec == [0.0]


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical() -> None:
    v = [1.0, 0.0, 1.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_zero_vector() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# PromptCluster construction
# ---------------------------------------------------------------------------


def test_prompt_cluster_defaults() -> None:
    c = PromptCluster(name="Test", anchor_phrases=["foo bar baz"])
    assert c.complexity_delta == 0.0
    assert c.stakes_delta == 0.0
    assert c.weight == 0.5


def test_default_clusters_non_empty() -> None:
    assert len(_DEFAULT_CLUSTERS) >= 5


def test_default_cluster_names() -> None:
    names = {c.name for c in _DEFAULT_CLUSTERS}
    assert "Analysis" in names
    assert "Factual" in names
    assert "Risk" in names


# ---------------------------------------------------------------------------
# SemanticBridge
# ---------------------------------------------------------------------------


def test_semantic_bridge_finds_analysis_cluster() -> None:
    bridge = SemanticBridge()
    cluster, sim = bridge.find_cluster("evaluate and compare the architectural options systematically")
    assert cluster is not None
    assert cluster.name == "Analysis"
    assert sim > 0.0


def test_semantic_bridge_finds_risk_cluster() -> None:
    bridge = SemanticBridge()
    cluster, sim = bridge.find_cluster("production emergency security breach critical compliance")
    assert cluster is not None
    assert cluster.name == "Risk"


def test_semantic_bridge_below_threshold_returns_none() -> None:
    bridge = SemanticBridge(min_similarity=0.999)  # impossible threshold
    cluster, sim = bridge.find_cluster("hello world")
    assert cluster is None


def test_semantic_bridge_apply_raises_complexity() -> None:
    bridge = SemanticBridge()
    base_c = 0.40
    new_c, *_ = bridge.apply(
        "analyze evaluate compare examine the situation thoroughly",
        complexity=base_c, stakes=0.40, uncertainty=0.40, tool_relevance=0.40
    )
    # Analysis cluster has positive complexity_delta
    assert new_c >= base_c


def test_semantic_bridge_apply_no_match_returns_unchanged() -> None:
    bridge = SemanticBridge(min_similarity=0.999)
    c, s, u, tr, name, sim = bridge.apply(
        "xyzzy quux frobnicate", 0.5, 0.4, 0.45, 0.40
    )
    assert c == 0.5
    assert name == ""


def test_semantic_bridge_lazy_build() -> None:
    bridge = SemanticBridge()
    assert not bridge._built
    bridge._build()
    assert bridge._built
    assert len(bridge._vocab) > 0


def test_semantic_bridge_custom_clusters() -> None:
    custom = PromptCluster(
        name="Custom",
        anchor_phrases=["zork dungeon adventure game"],
        complexity_delta=0.30,
    )
    bridge = SemanticBridge(clusters=[custom], min_similarity=0.01)
    cluster, sim = bridge.find_cluster("zork dungeon")
    assert cluster is not None
    assert cluster.name == "Custom"


# ---------------------------------------------------------------------------
# PromptFeatures — new fields
# ---------------------------------------------------------------------------


def test_prompt_features_default_confidence() -> None:
    enc = PromptEncoder()
    f = enc.encode("hello world")
    assert f.encoder_confidence == 1.0
    assert f.semantic_cluster == ""


def test_dual_encoder_features_have_confidence() -> None:
    enc = DualPathEncoder()
    f = enc.encode("What is 2 + 2?", "math")
    assert 0.0 <= f.encoder_confidence <= 1.0


# ---------------------------------------------------------------------------
# DualPathEncoder — core encode behaviour
# ---------------------------------------------------------------------------


def test_dual_encoder_returns_prompt_features() -> None:
    enc = DualPathEncoder()
    f = enc.encode("hello", "general")
    assert isinstance(f, PromptFeatures)


def test_dual_encoder_features_in_unit_interval() -> None:
    enc = DualPathEncoder()
    f = enc.encode("comprehensive analysis of market trends with compliance risks", "legal")
    for attr in ("complexity", "stakes", "uncertainty", "source_confidence", "tool_relevance"):
        val = getattr(f, attr)
        assert 0.0 <= val <= 1.0, f"{attr}={val:.3f}"


def test_dual_encoder_novel_prompt_triggers_slow_path() -> None:
    enc = DualPathEncoder(confidence_threshold=0.50)
    # "Scrutinize" and "GDPR" are not in keyword regex but are in Risk cluster
    f = enc.encode("Scrutinize the GDPR compliance protocols", domain="legal")
    assert f.semantic_cluster != "", (
        f"Expected slow-path to activate (conf={f.encoder_confidence:.3f})"
    )


def test_dual_encoder_known_prompt_uses_fast_path() -> None:
    enc = DualPathEncoder(confidence_threshold=0.05)  # very low threshold
    f = enc.encode("Comprehensive step-by-step analysis comparing three architectures", "general")
    assert f.semantic_cluster == "", "Dense keyword prompt should stay on fast-path"


def test_dual_encoder_inherits_ema_corrections() -> None:
    enc = DualPathEncoder()
    for _ in range(15):
        enc.update_from_price_delta("legal", cost_delta=0.50)
    f = enc.encode("legal contract review", "legal")
    corr = enc.corrections().get("legal", {}).get("complexity")
    assert corr is not None and corr.delta > 0.0


def test_dual_encoder_freeze_base_does_not_block_ema() -> None:
    enc = DualPathEncoder()
    enc.freeze_base()
    enc.update_from_price_delta("test", cost_delta=0.40)
    corr = enc.corrections().get("test", {}).get("complexity")
    assert corr is not None and corr.delta > 0.0


def test_dual_encoder_fine_tune_updates_domain() -> None:
    enc = DualPathEncoder()
    interactions = [
        {"cost_delta": 0.40, "risk_delta": 0.10},
        {"cost_delta": 0.35, "asset_delta": -0.30},
        {"cost_delta": 0.50},
    ]
    enc.freeze_base()
    enc.fine_tune(domain="medical", interactions=interactions)
    corr = enc.corrections().get("medical", {}).get("complexity")
    assert corr is not None
    assert corr.n_updates == 3
    assert corr.delta > 0.0


def test_dual_encoder_to_brain_task_works() -> None:
    from manifold import BrainConfig, ManifoldBrain, default_tools
    enc = DualPathEncoder()
    f = enc.encode("Analyze security vulnerabilities in authentication module", "security")
    task = f.to_brain_task(time_pressure=0.7)
    assert task.domain == "security"
    brain = ManifoldBrain(BrainConfig(generations=2, population_size=12, grid_size=5), tools=default_tools())
    decision = brain.decide(task)
    assert decision.action in {
        "answer", "clarify", "retrieve", "verify", "use_tool", "delegate",
        "plan", "explore", "exploit", "wait", "escalate", "refuse", "stop",
    }


# ---------------------------------------------------------------------------
# freeze_base / fine_tune on base PromptEncoder
# ---------------------------------------------------------------------------


def test_prompt_encoder_freeze_base_sets_flag() -> None:
    enc = PromptEncoder()
    enc.freeze_base()
    assert hasattr(enc, "_frozen") and enc._frozen


def test_prompt_encoder_fine_tune_empty_interactions() -> None:
    enc = PromptEncoder()
    enc.fine_tune("empty_domain", interactions=[])
    assert enc.corrections().get("empty_domain") is None


def test_prompt_encoder_fine_tune_multiple_interactions() -> None:
    enc = PromptEncoder()
    interactions = [
        {"cost_delta": 0.30, "risk_delta": 0.20, "asset_delta": -0.40},
        {"cost_delta": 0.25, "action": "analyze"},
    ]
    enc.fine_tune("finance", interactions)
    domain_corrections = enc.corrections().get("finance", {})
    assert "complexity" in domain_corrections
    assert "uncertainty" in domain_corrections
    assert "stakes" in domain_corrections


# ---------------------------------------------------------------------------
# run_dual_encoder_suite
# ---------------------------------------------------------------------------


def test_dual_encoder_suite_returns_three_findings() -> None:
    report = run_dual_encoder_suite(seed=5000)
    assert len(report.findings) == 3


def test_dual_encoder_suite_has_summary() -> None:
    report = run_dual_encoder_suite(seed=5000)
    assert len(report.honest_summary) >= 3


def test_dual_encoder_suite_all_pass() -> None:
    report = run_dual_encoder_suite(seed=5000)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.4f}. "
            f"{finding.interpretation}"
        )
