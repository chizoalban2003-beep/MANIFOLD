"""Tests for MANIFOLD simulation behavior."""

from __future__ import annotations

from dataclasses import replace

from manifold.connectors import load_connector_events
from manifold.rules import compile_rulebook
from manifold.simulation import (
    ACTION_DETOUR_RECHARGE,
    ACTION_PAUSE_RECHARGE,
    LAYER_INFO,
    LAYER_PHYSICAL,
    ManifoldConfig,
    PhaseConfig,
    run_manifold,
)


def test_ontogeny_phase_spends_nonzero_energy() -> None:
    config = ManifoldConfig(seed=21)
    result = run_manifold(config=config)
    phase4_metrics = [m for m in result.metrics if m.phase == "phase_4_ontogeny"]
    assert phase4_metrics, "Expected ontogeny phase metrics."
    assert any(m.average_energy_spent > 0.0 for m in phase4_metrics)


def test_teacher_events_occur_in_teacher_phases() -> None:
    config = ManifoldConfig(seed=7)
    result = run_manifold(config=config)
    teacher_events = [
        m.teacher_event
        for m in result.metrics
        if m.phase in {"phase_3_teacher_flicker", "phase_4_ontogeny"}
    ]
    assert any(event is not None for event in teacher_events)


def test_population_constraints_hold_across_generations() -> None:
    config = ManifoldConfig(seed=99)
    result = run_manifold(config=config)
    assert len(result.metrics) == config.total_generations
    for metric in result.metrics:
        assert config.min_population <= metric.population_size <= config.max_population


def test_recharge_phase_has_positive_recharge_events() -> None:
    config = ManifoldConfig(seed=13)
    result = run_manifold(config=config)
    phase5_metrics = [m for m in result.metrics if m.phase == "phase_5_recharge_hierarchical"]
    assert phase5_metrics, "Expected recharge phase metrics."
    assert any(m.average_recharge_gained > 0.0 for m in phase5_metrics)
    assert any(m.recharge_event_rate > 0.0 for m in phase5_metrics)


def test_recharge_phase_uses_hierarchical_actions() -> None:
    config = ManifoldConfig(seed=17)
    result = run_manifold(config=config)
    phase5_metrics = [m for m in result.metrics if m.phase == "phase_5_recharge_hierarchical"]
    assert phase5_metrics, "Expected recharge phase metrics."
    detour_total = sum(m.action_counts.get(ACTION_DETOUR_RECHARGE, 0) for m in phase5_metrics)
    pause_total = sum(m.action_counts.get(ACTION_PAUSE_RECHARGE, 0) for m in phase5_metrics)
    assert detour_total + pause_total > 0


def test_rule_compiler_parses_penalties_and_options() -> None:
    rules = compile_rulebook(
        "if late_delivery then -£8.20 @target=0.15 @alpha=1.2 @min=0.7 @max=20"
    )
    assert len(rules) == 1
    rule = rules[0]
    assert rule.name == "late_delivery"
    assert rule.penalty == 8.2
    assert rule.target_rate == 0.15
    assert rule.alpha == 1.2
    assert rule.min_penalty == 0.7
    assert rule.max_penalty == 20.0


def test_connector_loader_reads_csv_events(tmp_path) -> None:
    event_file = tmp_path / "events.csv"
    event_file.write_text(
        "generation,layer,cell,delta,note\n"
        "5,info_noise,4,1.2,traffic feed\n"
        "8,physical_risk,2,0.7,incident\n",
        encoding="utf-8",
    )
    events = load_connector_events(event_file)
    assert len(events) == 2
    assert events[0].generation == 5
    assert events[0].layer == "info_noise"
    assert events[1].cell == 2


def test_multi_layer_coupling_increases_info_contribution() -> None:
    phase = PhaseConfig(
        name="production_only",
        generations=6,
        dual_niche=True,
        teacher_enabled=True,
        flicker_enabled=True,
        ontogeny_enabled=True,
        recharge_enabled=True,
        multi_layer_enabled=True,
        adaptive_rules_enabled=True,
        predator_auto_tuning=True,
        rule_targets_enabled=True,
        energy_budget=12.0,
    )
    config = ManifoldConfig(seed=31, phases=(phase,))
    result = run_manifold(config=config)
    assert result.metrics
    last = result.metrics[-1]
    assert last.layer_regret_contrib[LAYER_INFO] > 0.0
    assert last.layer_regret_contrib[LAYER_PHYSICAL] > 0.0


def test_adaptive_rule_penalty_changes_when_enabled() -> None:
    phase = PhaseConfig(
        name="adaptive_only",
        generations=8,
        dual_niche=True,
        ontogeny_enabled=True,
        adaptive_rules_enabled=True,
        energy_budget=14.0,
    )
    config = ManifoldConfig(seed=44, phases=(phase,))
    result = run_manifold(config=config)
    penalties = [m.rule_penalties["late_delivery"] for m in result.metrics]
    assert any(abs(penalties[idx] - penalties[0]) > 1e-9 for idx in range(1, len(penalties)))


def test_legacy_mode_without_multilayer_has_no_confidence_distribution() -> None:
    legacy_phase = PhaseConfig(name="legacy", generations=5)
    config = ManifoldConfig(seed=5, phases=(legacy_phase,))
    result = run_manifold(config=config)
    last = result.metrics[-1]
    assert last.confidence_distribution == {}


def test_predator_autotuning_adjusts_spawn_rate() -> None:
    phase = PhaseConfig(
        name="predator",
        generations=10,
        dual_niche=True,
        teacher_enabled=True,
        ontogeny_enabled=True,
        predator_auto_tuning=True,
        energy_budget=12.0,
    )
    config = ManifoldConfig(seed=58, phases=(phase,))
    result = run_manifold(config=config)
    spawn_rates = [m.predator_spawn_rate for m in result.metrics]
    assert any(abs(rate - spawn_rates[0]) > 1e-9 for rate in spawn_rates[1:])


def test_transfer_artifact_contains_neutrality_and_penalties() -> None:
    config = ManifoldConfig(seed=22)
    result = run_manifold(config=config)
    artifact = result.transfer_artifact
    assert "neutrality_layer" in artifact
    assert "rule_penalties" in artifact
    assert len(artifact["neutrality_layer"]) == 9
