"""Tests for Phase 48: Meta-Prompt Evolution (manifold/meta.py)."""

from __future__ import annotations

import pytest

from manifold.meta import ABTestingEngine, PromptGenome


# ---------------------------------------------------------------------------
# PromptGenome — basic attributes
# ---------------------------------------------------------------------------


class TestPromptGenomeBasic:
    def test_default_grid_scores(self) -> None:
        g = PromptGenome(prompt_id="p1", template="You are helpful.")
        assert g.grid_scores == [0.0, 0.0, 0.0, 0.0]

    def test_default_counters(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        assert g.trial_count == 0
        assert g.success_count == 0

    def test_success_rate_zero_trials(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        assert g.success_rate == 0.0

    def test_success_rate_all_success(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        g.record_outcome(success=True)
        g.record_outcome(success=True)
        assert g.success_rate == pytest.approx(1.0)

    def test_success_rate_half(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        g.record_outcome(success=True)
        g.record_outcome(success=False)
        assert g.success_rate == pytest.approx(0.5)

    def test_success_rate_all_failure(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        for _ in range(5):
            g.record_outcome(success=False)
        assert g.success_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PromptGenome — record_outcome with grid_delta
# ---------------------------------------------------------------------------


class TestPromptGenomeGridScores:
    def test_first_outcome_sets_grid_score(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        g.record_outcome(success=True, grid_delta=[0.8, 0.2, 0.6, 0.9])
        assert g.grid_scores[0] == pytest.approx(0.8)
        assert g.grid_scores[1] == pytest.approx(0.2)

    def test_running_average_two_trials(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        g.record_outcome(success=True, grid_delta=[1.0, 0.0, 0.0, 0.0])
        g.record_outcome(success=True, grid_delta=[0.0, 0.0, 0.0, 0.0])
        assert g.grid_scores[0] == pytest.approx(0.5)

    def test_wrong_grid_delta_length_raises(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        with pytest.raises(ValueError, match="exactly 4"):
            g.record_outcome(success=True, grid_delta=[1.0, 2.0])

    def test_no_grid_delta_does_not_change_scores(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        g.record_outcome(success=True)
        assert g.grid_scores == [0.0, 0.0, 0.0, 0.0]

    def test_grid_scores_converge(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t")
        target = [0.5, 0.3, 0.7, 0.9]
        for _ in range(1000):
            g.record_outcome(success=True, grid_delta=target)
        for expected, actual in zip(target, g.grid_scores):
            assert actual == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# PromptGenome — mutate
# ---------------------------------------------------------------------------


class TestPromptGenomeMutate:
    def test_mutate_returns_new_genome(self) -> None:
        g = PromptGenome(prompt_id="base", template="You are a helpful agent.")
        m = g.mutate("child", seed=42)
        assert m is not g

    def test_mutate_different_template(self) -> None:
        g = PromptGenome(prompt_id="base", template="You are a helpful agent.")
        m = g.mutate("child", seed=42)
        assert m.template != g.template

    def test_mutate_inherits_grid_scores(self) -> None:
        g = PromptGenome(
            prompt_id="base", template="t", grid_scores=[0.5, 0.3, 0.7, 0.9]
        )
        m = g.mutate("child")
        assert m.grid_scores == pytest.approx([0.5, 0.3, 0.7, 0.9])

    def test_mutate_does_not_share_list(self) -> None:
        g = PromptGenome(
            prompt_id="base", template="t", grid_scores=[0.5, 0.3, 0.7, 0.9]
        )
        m = g.mutate("child")
        m.grid_scores[0] = 99.0
        assert g.grid_scores[0] == pytest.approx(0.5)

    def test_mutate_resets_trial_count(self) -> None:
        g = PromptGenome(prompt_id="base", template="t", trial_count=50, success_count=30)
        m = g.mutate("child")
        assert m.trial_count == 0
        assert m.success_count == 0

    def test_mutate_deterministic_with_seed(self) -> None:
        g = PromptGenome(prompt_id="base", template="The quick brown fox.")
        m1 = g.mutate("c1", seed=7)
        m2 = g.mutate("c2", seed=7)
        assert m1.template == m2.template

    def test_mutate_empty_template(self) -> None:
        g = PromptGenome(prompt_id="base", template="")
        m = g.mutate("child", seed=0)
        assert isinstance(m.template, str)


# ---------------------------------------------------------------------------
# PromptGenome — serialisation
# ---------------------------------------------------------------------------


class TestPromptGenomeSerialisation:
    def test_to_dict_keys(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t", trial_count=5, success_count=3)
        d = g.to_dict()
        assert set(d.keys()) == {
            "prompt_id",
            "template",
            "grid_scores",
            "trial_count",
            "success_count",
            "success_rate",
        }

    def test_to_dict_success_rate(self) -> None:
        g = PromptGenome(prompt_id="p1", template="t", trial_count=4, success_count=2)
        assert g.to_dict()["success_rate"] == pytest.approx(0.5)

    def test_from_dict_round_trip(self) -> None:
        original = PromptGenome(
            prompt_id="abc",
            template="hello world",
            grid_scores=[0.1, 0.2, 0.3, 0.4],
            trial_count=10,
            success_count=7,
        )
        rebuilt = PromptGenome.from_dict(original.to_dict())
        assert rebuilt.prompt_id == original.prompt_id
        assert rebuilt.template == original.template
        assert rebuilt.grid_scores == pytest.approx(original.grid_scores)
        assert rebuilt.trial_count == original.trial_count
        assert rebuilt.success_count == original.success_count


# ---------------------------------------------------------------------------
# ABTestingEngine — basic selection
# ---------------------------------------------------------------------------


class TestABTestingEngineSelect:
    def test_select_returns_champion_or_challenger(self) -> None:
        champion = PromptGenome(prompt_id="champ", template="Be a great agent.")
        engine = ABTestingEngine(champion=champion, seed=0)
        seen = set()
        for _ in range(50):
            g = engine.select()
            seen.add(g.prompt_id)
        # After 50 selections with 50/50 split both should appear
        assert "champ" in seen
        assert any("challenger" in pid for pid in seen)

    def test_select_champion_or_challenger_instance(self) -> None:
        champion = PromptGenome(prompt_id="c", template="t")
        engine = ABTestingEngine(champion=champion, seed=1)
        challenger = engine._ensure_challenger()
        for _ in range(30):
            g = engine.select()
            assert g is engine.champion or g is challenger


# ---------------------------------------------------------------------------
# ABTestingEngine — promotion logic
# ---------------------------------------------------------------------------


class TestABTestingEnginePromotion:
    def test_no_promotion_before_min_trials(self) -> None:
        champion = PromptGenome(
            prompt_id="champ", template="t", trial_count=100, success_count=60
        )
        engine = ABTestingEngine(champion=champion, min_trials=100, promotion_threshold=0.05, seed=0)
        challenger = engine._ensure_challenger()
        # Challenger has 5 successes / 5 trials → 100% but only 5 trials
        for _ in range(5):
            engine.record_outcome(challenger, success=True)
        assert engine.promotions == 0

    def test_promotion_occurs_when_criteria_met(self) -> None:
        champion = PromptGenome(
            prompt_id="champ", template="t", trial_count=200, success_count=100
        )  # 50% success rate
        engine = ABTestingEngine(champion=champion, min_trials=5, promotion_threshold=0.05, seed=0)
        challenger = engine._ensure_challenger()
        # Give challenger 8 successes / 8 trials → 100% > 55%
        for _ in range(8):
            engine.record_outcome(challenger, success=True)
        assert engine.promotions == 1

    def test_champion_replaced_after_promotion(self) -> None:
        champion = PromptGenome(
            prompt_id="champ", template="t", trial_count=200, success_count=100
        )
        engine = ABTestingEngine(champion=champion, min_trials=5, promotion_threshold=0.05, seed=0)
        challenger_before = engine._ensure_challenger()
        for _ in range(10):
            engine.record_outcome(challenger_before, success=True)
        assert engine.champion is not champion  # champion was replaced
        assert engine.champion.prompt_id == challenger_before.prompt_id

    def test_challenger_reset_after_promotion(self) -> None:
        champion = PromptGenome(
            prompt_id="champ", template="t", trial_count=200, success_count=100
        )
        engine = ABTestingEngine(champion=champion, min_trials=5, promotion_threshold=0.05, seed=0)
        challenger = engine._ensure_challenger()
        for _ in range(10):
            engine.record_outcome(challenger, success=True)
        # After promotion challenger slot is reset; a new one will be created lazily
        assert engine.challenger is None

    def test_no_promotion_when_challenger_worse(self) -> None:
        champion = PromptGenome(
            prompt_id="champ", template="t", trial_count=200, success_count=180
        )  # 90% success
        engine = ABTestingEngine(champion=champion, min_trials=5, promotion_threshold=0.05, seed=0)
        challenger = engine._ensure_challenger()
        for _ in range(10):
            engine.record_outcome(challenger, success=False)  # 0% success
        assert engine.promotions == 0

    def test_force_promote(self) -> None:
        champion = PromptGenome(
            prompt_id="champ", template="t", trial_count=200, success_count=100
        )
        engine = ABTestingEngine(
            champion=champion, min_trials=1000, promotion_threshold=0.05, seed=0
        )
        challenger = engine._ensure_challenger()
        # Only 3 trials but very high success rate
        for _ in range(3):
            engine.record_outcome(challenger, success=True)
        # Normal promote fails (below min_trials)
        assert engine.promotions == 0
        # Force promote
        promoted = engine.force_promote()
        assert promoted is True
        assert engine.promotions == 1

    def test_multiple_promotions(self) -> None:
        champion = PromptGenome(prompt_id="gen0", template="Base template.")
        engine = ABTestingEngine(champion=champion, min_trials=5, promotion_threshold=0.0, seed=42)
        for generation in range(3):
            challenger = engine._ensure_challenger()
            for _ in range(6):
                engine.record_outcome(challenger, success=True)
        assert engine.promotions >= 1


# ---------------------------------------------------------------------------
# ABTestingEngine — summary
# ---------------------------------------------------------------------------


class TestABTestingEngineSummary:
    def test_summary_keys(self) -> None:
        engine = ABTestingEngine(
            champion=PromptGenome(prompt_id="c", template="t"),
            min_trials=10,
        )
        s = engine.summary()
        assert "champion" in s
        assert "challenger" in s
        assert "promotions" in s
        assert "promotion_threshold" in s
        assert "min_trials" in s

    def test_summary_champion_is_dict(self) -> None:
        engine = ABTestingEngine(
            champion=PromptGenome(prompt_id="c", template="t"),
        )
        s = engine.summary()
        assert isinstance(s["champion"], dict)

    def test_summary_challenger_none_before_first_select(self) -> None:
        engine = ABTestingEngine(
            champion=PromptGenome(prompt_id="c", template="t"),
        )
        # _challenger is lazily created on first select
        assert engine.summary()["challenger"] is None

    def test_summary_challenger_populated_after_select(self) -> None:
        engine = ABTestingEngine(
            champion=PromptGenome(prompt_id="c", template="t"),
            seed=0,
        )
        engine.select()
        assert engine.summary()["challenger"] is not None

    def test_promotions_count_in_summary(self) -> None:
        champion = PromptGenome(
            prompt_id="c", template="t", trial_count=200, success_count=100
        )
        engine = ABTestingEngine(champion=champion, min_trials=5, promotion_threshold=0.05, seed=0)
        challenger = engine._ensure_challenger()
        for _ in range(10):
            engine.record_outcome(challenger, success=True)
        assert engine.summary()["promotions"] == 1


# ---------------------------------------------------------------------------
# ABTestingEngine — record_outcome with grid_delta
# ---------------------------------------------------------------------------


class TestABTestingEngineGridDelta:
    def test_record_outcome_passes_grid_delta(self) -> None:
        champion = PromptGenome(prompt_id="c", template="t")
        engine = ABTestingEngine(champion=champion, seed=0)
        engine.record_outcome(champion, success=True, grid_delta=[1.0, 0.5, 0.8, 0.9])
        assert champion.grid_scores[0] == pytest.approx(1.0)

    def test_record_outcome_returns_bool(self) -> None:
        champion = PromptGenome(prompt_id="c", template="t")
        engine = ABTestingEngine(champion=champion, seed=0)
        result = engine.record_outcome(champion, success=True)
        assert isinstance(result, bool)
