"""Tests for MANIFOLD simulation behavior."""

from __future__ import annotations

from manifold.simulation import (
    ACTION_DETOUR_RECHARGE,
    ACTION_PAUSE_RECHARGE,
    ManifoldConfig,
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
