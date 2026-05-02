"""Tests for Phase 6: ReputationRegistry, WarmStartConfig, warm_start_memory, and Phase 7 telemetry probes."""

import dataclasses

import pytest

from manifold import (
    BrainConfig,
    BrainMemory,
    BrainOutcome,
    BrainTask,
    GossipBus,
    LiveBrain,
    ManifoldBrain,
    ReputationRegistry,
    ToolProfile,
    WarmStartConfig,
    default_tools,
    warm_start_memory,
    run_server_telemetry_suite,
    run_warm_start_suite,
)


_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)
_TOOLS = default_tools()
_WEB_SEARCH = next(t for t in _TOOLS if t.name == "web_search")


# ---------------------------------------------------------------------------
# ReputationRegistry — basic API
# ---------------------------------------------------------------------------


def test_registry_initially_unknown() -> None:
    reg = ReputationRegistry()
    assert reg.global_success_rate("any_tool") is None


def test_registry_observe_single() -> None:
    reg = ReputationRegistry()
    reg.observe("tool_a", success_rate=0.70, n_observations=10)
    rate = reg.global_success_rate("tool_a")
    assert rate is not None
    assert abs(rate - 0.70) < 0.01


def test_registry_observe_multiple_batches_weighted() -> None:
    reg = ReputationRegistry()
    reg.observe("tool_b", success_rate=1.0, n_observations=10)
    reg.observe("tool_b", success_rate=0.0, n_observations=10)
    rate = reg.global_success_rate("tool_b")
    assert rate is not None
    # weighted average should be ~0.5
    assert abs(rate - 0.50) < 0.05


def test_registry_observation_count() -> None:
    reg = ReputationRegistry()
    reg.observe("tool_c", success_rate=0.8, n_observations=15)
    assert reg.observation_count("tool_c") == 15


def test_registry_observation_count_unknown_zero() -> None:
    reg = ReputationRegistry()
    assert reg.observation_count("unknown") == 0


def test_registry_all_rates_returns_copy() -> None:
    reg = ReputationRegistry()
    reg.observe("tool_d", success_rate=0.60, n_observations=5)
    rates = reg.all_rates()
    rates["injected"] = 0.99
    assert "injected" not in reg.all_rates()


def test_registry_observe_from_memory() -> None:
    reg = ReputationRegistry()
    memory = BrainMemory()
    memory.tool_stats["web_search"] = {
        "success_rate": 0.55, "count": 20.0, "utility": 0.6, "consecutive_failures": 0.0
    }
    memory.tool_stats["calculator"] = {
        "success_rate": 0.92, "count": 8.0, "utility": 0.9, "consecutive_failures": 0.0
    }
    reg.observe_from_memory(memory)

    ws = reg.global_success_rate("web_search")
    calc = reg.global_success_rate("calculator")
    assert ws is not None and abs(ws - 0.55) < 0.06
    assert calc is not None and abs(calc - 0.92) < 0.06


def test_registry_observe_from_empty_memory() -> None:
    reg = ReputationRegistry()
    reg.observe_from_memory(BrainMemory())  # must not raise


def test_registry_clamped_to_unit_interval() -> None:
    reg = ReputationRegistry()
    reg.observe("x", success_rate=1.5, n_observations=1)  # illegal but safe
    rate = reg.global_success_rate("x")
    # The registry stores what it's given — clamping is done in warm_start_memory
    assert rate is not None


# ---------------------------------------------------------------------------
# WarmStartConfig
# ---------------------------------------------------------------------------


def test_warm_start_config_related() -> None:
    cfg = WarmStartConfig(
        related_domains=frozenset({("support", "billing")}),
        related_alpha=0.8,
        unrelated_alpha=0.3,
    )
    assert cfg.alpha("support", "billing") == pytest.approx(0.8)


def test_warm_start_config_symmetric() -> None:
    cfg = WarmStartConfig(
        related_domains=frozenset({("support", "billing")}),
        related_alpha=0.8,
        unrelated_alpha=0.3,
    )
    assert cfg.alpha("billing", "support") == pytest.approx(0.8)


def test_warm_start_config_unrelated() -> None:
    cfg = WarmStartConfig(
        related_domains=frozenset({("support", "billing")}),
        related_alpha=0.8,
        unrelated_alpha=0.3,
    )
    assert cfg.alpha("support", "coding") == pytest.approx(0.3)


def test_warm_start_config_empty_defaults_unrelated() -> None:
    cfg = WarmStartConfig()
    assert cfg.alpha("x", "y") == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# warm_start_memory
# ---------------------------------------------------------------------------


def test_warm_start_memory_unknown_tool_skipped() -> None:
    reg = ReputationRegistry()
    # Only register calculator, not web_search
    reg.observe("calculator", success_rate=0.90, n_observations=10)
    memory = warm_start_memory(reg, _TOOLS, alpha=0.5)
    # web_search not in registry → not in warm memory
    assert "web_search" not in memory.tool_stats


def test_warm_start_memory_formula_correct() -> None:
    """Rep_0 = α × Rep_Global + (1-α) × Rep_Default"""
    alpha = 0.6
    global_rep = 0.40
    expected = alpha * global_rep + (1.0 - alpha) * _WEB_SEARCH.reliability

    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=global_rep, n_observations=20)
    memory = warm_start_memory(reg, _TOOLS, alpha=alpha)

    actual = memory.tool_stats["web_search"]["success_rate"]
    assert abs(actual - expected) < 0.02


def test_warm_start_memory_alpha_zero_uses_default() -> None:
    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=0.01, n_observations=20)
    memory = warm_start_memory(reg, _TOOLS, alpha=0.0)
    actual = memory.tool_stats["web_search"]["success_rate"]
    # alpha=0.0 → purely use tool.reliability
    assert abs(actual - _WEB_SEARCH.reliability) < 0.01


def test_warm_start_memory_alpha_one_uses_global() -> None:
    global_rep = 0.30
    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=global_rep, n_observations=20)
    memory = warm_start_memory(reg, _TOOLS, alpha=1.0)
    actual = memory.tool_stats["web_search"]["success_rate"]
    assert abs(actual - global_rep) < 0.01


def test_warm_start_memory_succeeds_rate_in_unit_interval() -> None:
    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=1.5, n_observations=5)  # out-of-range
    memory = warm_start_memory(reg, _TOOLS, alpha=0.5)
    actual = memory.tool_stats.get("web_search", {}).get("success_rate", 0.5)
    assert 0.0 <= actual <= 1.0


def test_warm_start_memory_count_preserved() -> None:
    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=0.65, n_observations=25)
    memory = warm_start_memory(reg, _TOOLS, alpha=0.5)
    count = memory.tool_stats["web_search"]["count"]
    assert count == 25.0


def test_warm_start_memory_lower_adj_than_cold_start() -> None:
    """Warm-started memory reports lower reliability for a known-bad tool."""
    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=0.30, n_observations=30)

    cold_memory = BrainMemory()
    warm_memory = warm_start_memory(reg, _TOOLS, alpha=0.7)

    cold_adj = cold_memory.tool_reliability_adjustment(_WEB_SEARCH)
    warm_adj = warm_memory.tool_reliability_adjustment(_WEB_SEARCH)

    assert warm_adj < cold_adj, (
        f"Warm start should inherit the scar: cold={cold_adj:.3f}, warm={warm_adj:.3f}"
    )


def test_warm_start_memory_used_by_manifold_brain() -> None:
    reg = ReputationRegistry()
    reg.observe("web_search", success_rate=0.20, n_observations=50)
    warm_memory = warm_start_memory(reg, _TOOLS, alpha=0.8)
    brain = ManifoldBrain(_CFG, tools=_TOOLS, memory=warm_memory)
    task = BrainTask("Search for compliance documents", "legal", tool_relevance=0.9, stakes=0.7)
    decision = brain.decide(task)
    assert decision.action is not None


# ---------------------------------------------------------------------------
# Phase 7: run_server_telemetry_suite
# ---------------------------------------------------------------------------


def test_server_telemetry_suite_returns_two_findings() -> None:
    report = run_server_telemetry_suite(seed=7000)
    assert len(report.findings) == 2
    assert {f.name for f in report.findings} == {
        "server_region_failure_reroute",
        "server_region_recovery_gossip",
    }


def test_server_telemetry_suite_all_pass() -> None:
    report = run_server_telemetry_suite(seed=7000)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.4f}. "
            f"{finding.interpretation}"
        )


def test_server_telemetry_suite_has_summary() -> None:
    report = run_server_telemetry_suite(seed=7000)
    assert len(report.honest_summary) >= 3


# ---------------------------------------------------------------------------
# Phase 6: run_warm_start_suite
# ---------------------------------------------------------------------------


def test_warm_start_suite_returns_four_findings() -> None:
    report = run_warm_start_suite(seed=6000)
    assert len(report.findings) == 4
    assert {f.name for f in report.findings} == {
        "warm_start_reduces_cold_start_regret",
        "reputation_transfer_alpha_scales",
        "warm_start_config_related_domains",
        "registry_observe_from_memory",
    }


def test_warm_start_suite_all_pass() -> None:
    report = run_warm_start_suite(seed=6000)
    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.4f}. "
            f"{finding.interpretation}"
        )


def test_warm_start_suite_has_summary() -> None:
    report = run_warm_start_suite(seed=6000)
    assert len(report.honest_summary) >= 4
