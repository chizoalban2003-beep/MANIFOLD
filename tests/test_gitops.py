"""Tests for Phase 19: GitOps Loop — CI/CD for Trust."""

from __future__ import annotations

import pytest

from manifold import (
    ConnectorRegistry,
    ToolConnector,
    ToolProfile,
)
from manifold.gitops import (
    AutonomousPRProposal,
    CIRiskDelta,
    ManifoldCICheck,
    PRDraft,
    generate_github_action,
    _proposals_to_rule_diffs,
    _render_yaml_diff,
)
from manifold.autodiscovery import AutoRuleDiscovery
from manifold.policy import ManifoldPolicy, PolicyDomain, RuleDiff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(name: str, risk: float = 0.20, reliability: float = 0.90) -> ToolProfile:
    return ToolProfile(
        name=name,
        cost=0.05,
        latency=0.3,
        reliability=reliability,
        risk=risk,
        asset=0.80,
        domain="general",
    )


def _registry_with(*profiles: ToolProfile) -> ConnectorRegistry:
    reg = ConnectorRegistry()
    for p in profiles:
        reg.register(ToolConnector(name=p.name, fn=lambda: None, profile=p))
    return reg


def _policy_with_domain(domain: str = "finance", risk_tol: float = 0.30, min_rel: float = 0.85) -> ManifoldPolicy:
    d = PolicyDomain(
        name=domain,
        stakes=0.9,
        risk_tolerance=risk_tol,
        coordination_tax_cap=0.10,
        fallback_strategy="hitl",
        min_tool_reliability=min_rel,
    )
    return ManifoldPolicy(domains=[d])


# ---------------------------------------------------------------------------
# CIRiskDelta
# ---------------------------------------------------------------------------


class TestCIRiskDelta:
    def test_fields_accessible(self) -> None:
        d = CIRiskDelta(
            tool_name="tool_a",
            old_risk=0.20,
            new_risk=0.30,
            delta=0.10,
            old_reliability=0.90,
            new_reliability=0.85,
            reliability_delta=-0.05,
            vetoed=False,
        )
        assert d.tool_name == "tool_a"
        assert d.delta == pytest.approx(0.10)
        assert not d.vetoed

    def test_vetoed_flag(self) -> None:
        d = CIRiskDelta(
            tool_name="risky",
            old_risk=0.40,
            new_risk=0.60,
            delta=0.20,
            old_reliability=0.80,
            new_reliability=0.70,
            reliability_delta=-0.10,
            vetoed=True,
        )
        assert d.vetoed

    def test_new_tool_none_old(self) -> None:
        d = CIRiskDelta(
            tool_name="new_tool",
            old_risk=None,
            new_risk=0.25,
            delta=0.0,
            old_reliability=None,
            new_reliability=0.88,
            reliability_delta=0.0,
            vetoed=False,
        )
        assert d.old_risk is None
        assert d.new_risk == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# ManifoldCICheck — profile-level checks
# ---------------------------------------------------------------------------


class TestManifoldCICheckProfiles:
    def test_no_change_passes(self) -> None:
        profiles = [_profile("tool_a"), _profile("tool_b")]
        checker = ManifoldCICheck()
        report = checker.check(profiles, profiles)
        assert report.passed
        assert len(report.deltas) == 2

    def test_risk_spike_fails(self) -> None:
        old_p = _profile("tool_a", risk=0.20)
        new_p = _profile("tool_a", risk=0.60)
        checker = ManifoldCICheck(risk_veto_threshold=0.45)
        report = checker.check([old_p], [new_p])
        assert not report.passed
        assert "tool_a" in report.vetoed_tools

    def test_risk_below_threshold_passes(self) -> None:
        old_p = _profile("tool_a", risk=0.30)
        new_p = _profile("tool_a", risk=0.40)
        checker = ManifoldCICheck(risk_veto_threshold=0.45, max_risk_delta=0.15)
        report = checker.check([old_p], [new_p])
        assert report.passed

    def test_risk_delta_exceeded_fails(self) -> None:
        old_p = _profile("tool_a", risk=0.20)
        new_p = _profile("tool_a", risk=0.38)
        checker = ManifoldCICheck(max_risk_delta=0.15)
        report = checker.check([old_p], [new_p])
        assert not report.passed

    def test_reliability_drop_fails(self) -> None:
        old_p = _profile("tool_a", reliability=0.90)
        new_p = _profile("tool_a", reliability=0.70)
        checker = ManifoldCICheck(max_reliability_drop=0.10)
        report = checker.check([old_p], [new_p])
        assert not report.passed

    def test_new_tool_not_vetoed_by_default(self) -> None:
        new_p = _profile("brand_new", risk=0.30)
        checker = ManifoldCICheck()
        report = checker.check([], [new_p])
        assert report.passed
        assert len(report.deltas) == 1

    def test_multiple_tools_one_veto(self) -> None:
        old = [_profile("a", risk=0.10), _profile("b", risk=0.20)]
        new = [_profile("a", risk=0.10), _profile("b", risk=0.55)]
        checker = ManifoldCICheck(risk_veto_threshold=0.45)
        report = checker.check(old, new)
        assert not report.passed
        assert "b" in report.vetoed_tools
        assert "a" not in report.vetoed_tools

    def test_summary_text_passed(self) -> None:
        profiles = [_profile("t")]
        checker = ManifoldCICheck()
        report = checker.check(profiles, profiles)
        assert "PASSED" in report.summary

    def test_summary_text_failed(self) -> None:
        old_p = _profile("t", risk=0.10)
        new_p = _profile("t", risk=0.60)
        checker = ManifoldCICheck()
        report = checker.check([old_p], [new_p])
        assert "FAILED" in report.summary

    def test_empty_new_profiles(self) -> None:
        old = [_profile("t")]
        checker = ManifoldCICheck()
        report = checker.check(old, [])
        assert report.passed  # No new profiles = nothing to veto
        assert len(report.deltas) == 0


# ---------------------------------------------------------------------------
# ManifoldCICheck — policy-driven threshold
# ---------------------------------------------------------------------------


class TestManifoldCICheckPolicy:
    def test_policy_overrides_threshold(self) -> None:
        policy = _policy_with_domain(risk_tol=0.25)
        checker = ManifoldCICheck(policy=policy)
        assert checker.risk_veto_threshold == pytest.approx(0.25)

    def test_no_policy_uses_default(self) -> None:
        checker = ManifoldCICheck()
        assert checker.risk_veto_threshold == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# ManifoldCICheck — registry check
# ---------------------------------------------------------------------------


class TestManifoldCICheckRegistry:
    def test_registry_check_passes(self) -> None:
        p = _profile("t", risk=0.20)
        reg = _registry_with(p)
        checker = ManifoldCICheck()
        report = checker.check_registry(reg, reg)
        assert report.passed

    def test_registry_check_fails_on_risk_spike(self) -> None:
        old_p = _profile("t", risk=0.10)
        new_p = _profile("t", risk=0.60)
        old_reg = _registry_with(old_p)
        new_reg = _registry_with(new_p)
        checker = ManifoldCICheck(risk_veto_threshold=0.45)
        report = checker.check_registry(old_reg, new_reg)
        assert not report.passed


# ---------------------------------------------------------------------------
# ManifoldCICheck — policy diff check
# ---------------------------------------------------------------------------


class TestManifoldCICheckPolicyDiff:
    def test_pending_diff_risk_spike_fails(self) -> None:
        diff = RuleDiff(
            rule_name="fraud_block",
            trigger="high_risk",
            domain="finance",
            field="risk_tolerance",
            old_value=0.30,
            new_value=0.60,  # spike above 0.45
            rationale="test",
            confidence=0.9,
            status="pending",
        )
        policy = ManifoldPolicy(pending_diffs=[diff])
        checker = ManifoldCICheck(risk_veto_threshold=0.45)
        report = checker.check_policy_diff(policy)
        assert not report.passed

    def test_pending_diff_normal_change_passes(self) -> None:
        diff = RuleDiff(
            rule_name="mild_rule",
            trigger="low_risk",
            domain="finance",
            field="risk_tolerance",
            old_value=0.30,
            new_value=0.35,
            rationale="test",
            confidence=0.9,
            status="pending",
        )
        policy = ManifoldPolicy(pending_diffs=[diff])
        checker = ManifoldCICheck(risk_veto_threshold=0.45)
        report = checker.check_policy_diff(policy)
        assert report.passed

    def test_approved_diff_not_checked(self) -> None:
        diff = RuleDiff(
            rule_name="approved",
            trigger="x",
            domain="finance",
            field="risk_tolerance",
            old_value=0.30,
            new_value=0.80,  # huge spike, but already approved
            rationale="test",
            confidence=0.9,
            status="approved",
        )
        policy = ManifoldPolicy(pending_diffs=[diff])
        checker = ManifoldCICheck(risk_veto_threshold=0.45)
        report = checker.check_policy_diff(policy)
        assert report.passed  # approved diffs are skipped


# ---------------------------------------------------------------------------
# CIRiskReport.as_markdown
# ---------------------------------------------------------------------------


class TestCIRiskReportMarkdown:
    def test_passed_markdown(self) -> None:
        profiles = [_profile("tool_a")]
        checker = ManifoldCICheck()
        report = checker.check(profiles, profiles)
        md = report.as_markdown()
        assert "PASSED" in md
        assert "tool_a" in md

    def test_failed_markdown(self) -> None:
        old_p = _profile("bad_tool", risk=0.10)
        new_p = _profile("bad_tool", risk=0.70)
        checker = ManifoldCICheck()
        report = checker.check([old_p], [new_p])
        md = report.as_markdown()
        assert "FAILED" in md
        assert "bad_tool" in md
        assert "VETO" in md

    def test_empty_deltas_markdown(self) -> None:
        checker = ManifoldCICheck()
        report = checker.check([], [])
        md = report.as_markdown()
        assert "No tool profiles changed" in md


# ---------------------------------------------------------------------------
# AutonomousPRProposal
# ---------------------------------------------------------------------------


def _discovery_with_observations() -> AutoRuleDiscovery:
    ard = AutoRuleDiscovery()
    from manifold.brain import BrainOutcome
    outcome = BrainOutcome(
        success=False,
        cost_paid=0.05,
        risk_realized=0.45,
        asset_gained=-0.45,
    )
    for _ in range(6):
        ard.observe_rule_event(
            rule_name="stakes_penalty",
            trigger="high_stakes",
            outcome=outcome,
            penalty=0.20,
        )
    return ard


class TestAutonomousPRProposal:
    def test_draft_returns_pr_draft(self) -> None:
        ard = _discovery_with_observations()
        policy = ManifoldPolicy.from_template("finance")
        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        pr = proposer.draft()
        assert isinstance(pr, PRDraft)

    def test_pr_has_title_and_body(self) -> None:
        ard = _discovery_with_observations()
        policy = ManifoldPolicy.from_template("finance")
        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        pr = proposer.draft()
        assert len(pr.title) > 0
        assert len(pr.body) > 0

    def test_body_contains_author(self) -> None:
        ard = _discovery_with_observations()
        policy = ManifoldPolicy()
        proposer = AutonomousPRProposal(discovery=ard, policy=policy, author="ci-bot")
        pr = proposer.draft()
        assert "ci-bot" in pr.body

    def test_no_proposals_title(self) -> None:
        ard = AutoRuleDiscovery()  # no observations
        policy = ManifoldPolicy()
        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        pr = proposer.draft()
        assert "No penalty changes" in pr.title

    def test_min_confidence_filters(self) -> None:
        ard = _discovery_with_observations()
        policy = ManifoldPolicy()
        proposer = AutonomousPRProposal(
            discovery=ard, policy=policy, min_confidence=1.0  # impossible
        )
        pr = proposer.draft()
        assert len(pr.proposals) == 0

    def test_drafts_accumulate(self) -> None:
        ard = _discovery_with_observations()
        policy = ManifoldPolicy()
        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        proposer.draft()
        proposer.draft()
        assert len(proposer.drafts()) == 2

    def test_draft_from_diffs(self) -> None:
        ard = AutoRuleDiscovery()
        policy = ManifoldPolicy.from_template("finance")
        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        diff = RuleDiff(
            rule_name="test_rule",
            trigger="test_trigger",
            domain="finance",
            field="risk_tolerance",
            old_value=0.30,
            new_value=0.20,
            rationale="test",
            confidence=0.8,
        )
        pr = proposer.draft_from_diffs([diff])
        assert "test_rule" in pr.body
        assert len(pr.diffs) == 1

    def test_body_contains_yaml_diff(self) -> None:
        ard = _discovery_with_observations()
        policy = ManifoldPolicy()
        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        pr = proposer.draft()
        assert "yaml" in pr.body.lower() or "YAML" in pr.body or "#" in pr.yaml_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestGitopsHelpers:
    def test_proposals_to_rule_diffs_empty(self) -> None:
        diffs = _proposals_to_rule_diffs([])
        assert diffs == []

    def test_render_yaml_diff_empty(self) -> None:
        policy = ManifoldPolicy()
        diff_str = _render_yaml_diff(policy, [])
        assert "No changes" in diff_str

    def test_render_yaml_diff_with_diff(self) -> None:
        policy = ManifoldPolicy.from_template("finance")
        diff = RuleDiff(
            rule_name="fraud",
            trigger="miss",
            domain="finance",
            field="risk_tolerance",
            old_value=0.30,
            new_value=0.25,
            rationale="observed loss",
            confidence=0.85,
        )
        result = _render_yaml_diff(policy, [diff])
        assert "fraud" in result
        assert "0.30" in result or "0.3" in result


# ---------------------------------------------------------------------------
# generate_github_action
# ---------------------------------------------------------------------------


class TestGenerateGithubAction:
    def test_returns_string(self) -> None:
        yaml = generate_github_action()
        assert isinstance(yaml, str)
        assert len(yaml) > 100

    def test_contains_workflow_key(self) -> None:
        yaml = generate_github_action()
        assert "name:" in yaml
        assert "manifold" in yaml.lower()

    def test_contains_python_step(self) -> None:
        yaml = generate_github_action()
        assert "python" in yaml.lower()
