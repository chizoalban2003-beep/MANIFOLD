"""Tests for Phase 27: Braintrust Consensus (manifold/consensus.py)."""

from __future__ import annotations

import pytest

from manifold.brain import BrainConfig, BrainTask
from manifold.consensus import (
    Braintrust,
    ConsensusResult,
    SocialGenome,
)
from manifold.interceptor import ActiveInterceptor, InterceptorConfig
from manifold.connector import ConnectorRegistry, ToolConnector
from manifold.brain import ManifoldBrain, ToolProfile


# ---------------------------------------------------------------------------
# SocialGenome tests
# ---------------------------------------------------------------------------


class TestSocialGenome:
    def test_risk_averse_factory(self) -> None:
        g = SocialGenome.risk_averse()
        assert g.name == "risk_averse"
        assert isinstance(g.config, BrainConfig)
        assert g.config.predators > 0.05  # more adversarial pressure

    def test_asset_aggressive_factory(self) -> None:
        g = SocialGenome.asset_aggressive()
        assert g.name == "asset_aggressive"
        assert g.config.predators < 0.05  # less pressure

    def test_balanced_factory(self) -> None:
        g = SocialGenome.balanced()
        assert g.name == "balanced"
        assert g.config.predators == 0.05

    def test_vote_weights_positive(self) -> None:
        for genome in [
            SocialGenome.risk_averse(),
            SocialGenome.asset_aggressive(),
            SocialGenome.balanced(),
        ]:
            assert genome.vote_weight > 0

    def test_risk_averse_vote_weight_higher(self) -> None:
        ra = SocialGenome.risk_averse()
        ba = SocialGenome.balanced()
        assert ra.vote_weight >= ba.vote_weight

    def test_frozen_dataclass(self) -> None:
        g = SocialGenome.balanced()
        with pytest.raises((AttributeError, TypeError)):
            g.name = "override"  # type: ignore[misc]

    def test_different_seeds(self) -> None:
        ra = SocialGenome.risk_averse()
        aa = SocialGenome.asset_aggressive()
        ba = SocialGenome.balanced()
        seeds = {ra.config.seed, aa.config.seed, ba.config.seed}
        # All three should have distinct seeds
        assert len(seeds) == 3


# ---------------------------------------------------------------------------
# ConsensusVote tests
# ---------------------------------------------------------------------------


class TestConsensusVote:
    def _make_brain_task(self) -> BrainTask:
        return BrainTask(prompt="test", domain="general", stakes=0.3)

    def test_approves_flag_reflects_action(self) -> None:
        bt = Braintrust()
        task = self._make_brain_task()
        result = bt.evaluate(task)
        for vote in result.votes:
            veto_actions = {"refuse", "escalate"}
            if vote.decision.action in veto_actions:
                assert not vote.approves
            else:
                assert vote.approves

    def test_weighted_confidence_bounded(self) -> None:
        bt = Braintrust()
        task = self._make_brain_task()
        result = bt.evaluate(task)
        for vote in result.votes:
            assert 0.0 <= vote.weighted_confidence <= 1.0

    def test_vote_weight_sums_to_approx_1(self) -> None:
        bt = Braintrust()
        task = self._make_brain_task()
        result = bt.evaluate(task)
        total_w = sum(v.vote_weight for v in result.votes)
        assert total_w == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Braintrust tests
# ---------------------------------------------------------------------------


class TestBraintrustBasics:
    def test_default_panel_size(self) -> None:
        bt = Braintrust()
        assert bt.panel_summary()["panel_size"] == 3

    def test_panel_genomes_named(self) -> None:
        bt = Braintrust()
        genomes = bt.panel_summary()["genomes"]
        assert "risk_averse" in genomes
        assert "asset_aggressive" in genomes
        assert "balanced" in genomes

    def test_evaluate_returns_consensus_result(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="What is 2+2?", domain="math", stakes=0.1)
        result = bt.evaluate(task)
        assert isinstance(result, ConsensusResult)

    def test_votes_count_equals_panel_size(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="test", domain="general", stakes=0.2)
        result = bt.evaluate(task)
        assert len(result.votes) == 3

    def test_consensus_score_bounded(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="test", domain="general", stakes=0.5)
        result = bt.evaluate(task)
        assert 0.0 <= result.consensus_score <= 1.0

    def test_threshold_from_config(self) -> None:
        config = InterceptorConfig(risk_veto_threshold=0.4)
        bt = Braintrust(config=config)
        task = BrainTask(prompt="test", domain="general", stakes=0.3)
        result = bt.evaluate(task)
        assert result.threshold == pytest.approx(1.0 - 0.4, abs=1e-6)

    def test_approved_consistent_with_score(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="safe action", domain="general", stakes=0.1)
        result = bt.evaluate(task)
        if result.consensus_score >= result.threshold:
            assert result.approved
        else:
            assert not result.approved

    def test_reason_not_empty(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="test", domain="general", stakes=0.3)
        result = bt.evaluate(task)
        assert result.reason

    def test_winning_action_is_string(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="test", domain="general", stakes=0.3)
        result = bt.evaluate(task)
        assert isinstance(result.winning_action, str)
        assert result.winning_action

    def test_task_stored_in_result(self) -> None:
        bt = Braintrust()
        task = BrainTask(prompt="unique_prompt_xyz", domain="finance", stakes=0.7)
        result = bt.evaluate(task)
        assert result.task is task

    def test_low_stakes_more_likely_approved(self) -> None:
        bt = Braintrust()
        low_task = BrainTask(prompt="low risk", domain="general", stakes=0.05, uncertainty=0.1)
        high_task = BrainTask(prompt="high risk", domain="general", stakes=0.95, uncertainty=0.9)
        low_result = bt.evaluate(low_task)
        high_result = bt.evaluate(high_task)
        # Low-stakes task should have higher or equal consensus score
        assert low_result.consensus_score >= high_result.consensus_score

    def test_custom_genomes(self) -> None:
        genomes = [SocialGenome.balanced(), SocialGenome.balanced()]
        bt = Braintrust(genomes=genomes)
        assert bt.panel_summary()["panel_size"] == 2

    def test_panel_summary_has_threshold(self) -> None:
        bt = Braintrust()
        summary = bt.panel_summary()
        assert "threshold" in summary
        assert 0.0 <= summary["threshold"] <= 1.0

    def test_evaluate_multiple_tasks_independent(self) -> None:
        bt = Braintrust()
        t1 = BrainTask(prompt="task1", domain="math", stakes=0.2)
        t2 = BrainTask(prompt="task2", domain="finance", stakes=0.8)
        r1 = bt.evaluate(t1)
        r2 = bt.evaluate(t2)
        # Results are independent objects
        assert r1 is not r2

    def test_refused_task_winning_action_is_refuse(self) -> None:
        """Very high stakes should push at least some brains toward refuse."""
        bt = Braintrust()
        task = BrainTask(
            prompt="very dangerous action",
            domain="general",
            stakes=1.0,
            uncertainty=1.0,
            safety_sensitivity=1.0,
        )
        result = bt.evaluate(task)
        # If not approved, winning action must be refuse
        if not result.approved:
            assert result.winning_action == "refuse"


# ---------------------------------------------------------------------------
# ActiveInterceptor consensus_mode tests
# ---------------------------------------------------------------------------


class TestActiveInterceptorConsensusMode:
    def _make_interceptor(
        self,
        *,
        consensus_mode: bool = False,
        risk_veto_threshold: float = 0.45,
    ) -> tuple[ActiveInterceptor, ConnectorRegistry]:
        registry = ConnectorRegistry()
        profile = ToolProfile(
            name="safe_tool",
            cost=0.01,
            latency=0.1,
            reliability=0.95,
            risk=0.05,
            asset=0.90,
            domain="general",
        )
        connector = ToolConnector(
            name="safe_tool",
            fn=lambda: {"ok": True},
            profile=profile,
        )
        registry.register(connector)
        config = InterceptorConfig(
            risk_veto_threshold=risk_veto_threshold,
            redirect_strategy="hitl",
        )
        brain = ManifoldBrain(config=BrainConfig(), tools=[profile])
        interceptor = ActiveInterceptor(
            registry=registry,
            brain=brain,
            config=config,
            consensus_mode=consensus_mode,
        )
        return interceptor, registry

    def test_consensus_mode_false_by_default(self) -> None:
        interceptor, _ = self._make_interceptor()
        assert interceptor.consensus_mode is False

    def test_consensus_mode_true_flag(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=True)
        assert interceptor.consensus_mode is True

    def test_intercept_single_brain_mode(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=False)
        task = BrainTask(prompt="safe op", domain="general", stakes=0.1)
        result = interceptor.intercept(task, "safe_tool")
        assert result.tool_name == "safe_tool"

    def test_intercept_consensus_mode(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=True)
        task = BrainTask(prompt="safe op", domain="general", stakes=0.1)
        result = interceptor.intercept(task, "safe_tool")
        assert result.tool_name == "safe_tool"

    def test_consensus_mode_returns_intercept_result(self) -> None:
        from manifold.interceptor import InterceptResult
        interceptor, _ = self._make_interceptor(consensus_mode=True)
        task = BrainTask(prompt="test", domain="general", stakes=0.2)
        result = interceptor.intercept(task, "safe_tool")
        assert isinstance(result, InterceptResult)

    def test_braintrust_lazy_init(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=True)
        # _braintrust is None before first intercept
        assert interceptor._braintrust is None
        task = BrainTask(prompt="test", domain="general", stakes=0.2)
        interceptor.intercept(task, "safe_tool")
        # After first intercept, _braintrust is populated
        assert interceptor._braintrust is not None

    def test_missing_tool_raises_key_error_in_consensus_mode(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=True)
        task = BrainTask(prompt="test", domain="general", stakes=0.2)
        with pytest.raises(KeyError):
            interceptor.intercept(task, "nonexistent_tool")

    def test_switch_off_consensus_still_works(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=False)
        task = BrainTask(prompt="normal", domain="general", stakes=0.1)
        result = interceptor.intercept(task, "safe_tool")
        assert result is not None

    def test_veto_count_updated_in_consensus_mode(self) -> None:
        interceptor, _ = self._make_interceptor(consensus_mode=True)
        task = BrainTask(prompt="test", domain="general", stakes=0.1)
        interceptor.intercept(task, "safe_tool")
        # Log should have exactly 1 entry
        assert len(interceptor.intercept_log()) == 1
