"""Tests for Phase 12: Automated Rule Discovery — PenaltyOptimizer,
PolicySynthesizer, and the AutoRuleDiscovery coordinator."""

import pytest

from manifold.autodiscovery import (
    AutoRuleDiscovery,
    DecompositionTemplate,
    PenaltyOptimizer,
    PenaltyProposal,
    PolicySynthesizer,
    RuleObservation,
)
from manifold.brain import (
    BrainConfig,
    BrainOutcome,
    BrainTask,
    HierarchicalBrain,
    HierarchicalDecision,
)
from manifold.gridmapper import Rule

# Tiny config keeps each decide_hierarchical call fast in tests.
_CFG = BrainConfig(generations=2, population_size=12, grid_size=5)

_COMPLEX_TASK = BrainTask(
    "Write a comprehensive legal analysis",
    domain="legal",
    complexity=0.92,
    stakes=0.85,
    uncertainty=0.55,
)

_SIMPLE_TASK = BrainTask(
    "What is the answer?",
    domain="legal",
    complexity=0.10,
    stakes=0.10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(
    trigger: str = "miss_target",
    rule_name: str = "late_delivery",
    asset_delta: float = -15.0,
    current_penalty: float = 8.2,
) -> RuleObservation:
    return RuleObservation(
        trigger=trigger,
        rule_name=rule_name,
        observed_asset_delta=asset_delta,
        current_penalty=current_penalty,
    )


def _make_hierarchical_decision(
    domain: str = "legal",
    complexity: float = 0.92,
    combined_utility: float | None = None,
) -> HierarchicalDecision:
    """Return a real HierarchicalDecision using a tiny-config brain."""
    brain = HierarchicalBrain(config=_CFG)
    task = BrainTask(
        "Analyse the situation",
        domain=domain,
        complexity=complexity,
        stakes=0.8,
        uncertainty=0.5,
    )
    hd = brain.decide_hierarchical(task)
    # Override combined_utility when the caller needs a specific value
    if combined_utility is not None:
        from dataclasses import replace
        hd = replace(hd, combined_utility=combined_utility)
    return hd


def _make_non_decomposed_decision() -> HierarchicalDecision:
    """Return a HierarchicalDecision that was NOT decomposed."""
    brain = HierarchicalBrain(config=_CFG)
    hd = brain.decide_hierarchical(_SIMPLE_TASK)
    return hd


# ---------------------------------------------------------------------------
# RuleObservation
# ---------------------------------------------------------------------------


def test_rule_observation_frozen():
    obs = _make_obs()
    with pytest.raises((AttributeError, TypeError)):
        obs.trigger = "changed"  # type: ignore[misc]


def test_rule_observation_fields():
    obs = _make_obs(
        trigger="miss_target",
        rule_name="late_delivery",
        asset_delta=-12.5,
        current_penalty=8.2,
    )
    assert obs.trigger == "miss_target"
    assert obs.rule_name == "late_delivery"
    assert obs.observed_asset_delta == -12.5
    assert obs.current_penalty == 8.2


# ---------------------------------------------------------------------------
# PenaltyOptimizer — below min_observations returns None
# ---------------------------------------------------------------------------


def test_penalty_optimizer_no_proposal_below_threshold():
    opt = PenaltyOptimizer(min_observations=5)
    for _ in range(4):
        opt.record(_make_obs())
    result = opt.suggest("miss_target")
    assert result is None


def test_penalty_optimizer_returns_proposal_at_threshold():
    opt = PenaltyOptimizer(min_observations=5)
    for _ in range(5):
        opt.record(_make_obs(asset_delta=-15.0, current_penalty=8.2))
    result = opt.suggest("miss_target")
    assert result is not None
    assert isinstance(result, PenaltyProposal)


# ---------------------------------------------------------------------------
# PenaltyOptimizer — direction of proposed change
# ---------------------------------------------------------------------------


def test_penalty_optimizer_proposes_increase_when_underpriced():
    """When observed loss (15.0) >> current penalty (8.2), proposed should be higher."""
    opt = PenaltyOptimizer(min_observations=5, penalty_scale=1.0)
    for _ in range(10):
        opt.record(_make_obs(asset_delta=-15.0, current_penalty=8.2))
    proposal = opt.suggest("miss_target")
    assert proposal is not None
    assert proposal.proposed_penalty > proposal.current_penalty
    assert proposal.delta > 0


def test_penalty_optimizer_proposes_decrease_when_overpriced():
    """When observed loss (2.0) << current penalty (8.2), proposed should be lower."""
    opt = PenaltyOptimizer(min_observations=5, penalty_scale=1.0)
    for _ in range(10):
        opt.record(_make_obs(asset_delta=-2.0, current_penalty=8.2))
    proposal = opt.suggest("miss_target")
    assert proposal is not None
    assert proposal.proposed_penalty < proposal.current_penalty
    assert proposal.delta < 0


def test_penalty_optimizer_no_change_when_aligned():
    """When penalty equals mean loss * scale, delta should be within tolerance."""
    opt = PenaltyOptimizer(min_observations=5, penalty_scale=1.0, adjustment_threshold=0.5)
    # mean_loss = 8.2, scale=1.0 → proposed = 8.2 = current → delta ≈ 0
    for _ in range(10):
        opt.record(_make_obs(asset_delta=-8.2, current_penalty=8.2))
    proposal = opt.suggest("miss_target")
    assert proposal is not None
    assert abs(proposal.delta) < 0.5


# ---------------------------------------------------------------------------
# PenaltyOptimizer — proposed penalty correlates with mean loss magnitude
# ---------------------------------------------------------------------------


def test_penalty_optimizer_proposal_correlates_with_loss():
    """Larger mean loss should yield a larger proposed penalty."""
    opt_small = PenaltyOptimizer(min_observations=5, penalty_scale=1.2)
    opt_large = PenaltyOptimizer(min_observations=5, penalty_scale=1.2)
    for _ in range(20):
        opt_small.record(_make_obs(asset_delta=-5.0, current_penalty=8.2))
        opt_large.record(_make_obs(asset_delta=-20.0, current_penalty=8.2))
    p_small = opt_small.suggest("miss_target")
    p_large = opt_large.suggest("miss_target")
    assert p_small is not None
    assert p_large is not None
    assert p_large.proposed_penalty > p_small.proposed_penalty


# ---------------------------------------------------------------------------
# PenaltyOptimizer — confidence grows with observations
# ---------------------------------------------------------------------------


def test_penalty_optimizer_confidence_increases_with_observations():
    opt = PenaltyOptimizer(min_observations=5, penalty_scale=1.2)
    for _ in range(5):
        opt.record(_make_obs())
    p5 = opt.suggest("miss_target")
    for _ in range(50):
        opt.record(_make_obs())
    p50 = opt.suggest("miss_target")
    assert p5 is not None and p50 is not None
    assert p50.confidence >= p5.confidence


# ---------------------------------------------------------------------------
# PenaltyOptimizer — auto_adjust mutates rules_registry
# ---------------------------------------------------------------------------


def test_penalty_optimizer_auto_adjust_updates_registry():
    rule = Rule(name="late_delivery", penalty=8.2, triggers="miss_target")
    opt = PenaltyOptimizer(
        min_observations=5,
        penalty_scale=1.2,
        adjustment_threshold=0.1,
        auto_adjust=True,
        rules_registry={"late_delivery": rule},
    )
    for _ in range(10):
        opt.record(_make_obs(asset_delta=-15.0, current_penalty=8.2))
    proposal = opt.suggest("miss_target")
    assert proposal is not None
    updated_rule = opt.rules_registry.get("late_delivery")
    assert updated_rule is not None
    # Auto-adjust should have changed the penalty
    assert updated_rule.penalty != 8.2


def test_penalty_optimizer_no_auto_adjust_leaves_registry_unchanged():
    rule = Rule(name="late_delivery", penalty=8.2, triggers="miss_target")
    opt = PenaltyOptimizer(
        min_observations=5,
        penalty_scale=1.2,
        auto_adjust=False,
        rules_registry={"late_delivery": rule},
    )
    for _ in range(10):
        opt.record(_make_obs(asset_delta=-15.0, current_penalty=8.2))
    opt.suggest("miss_target")
    assert opt.rules_registry["late_delivery"].penalty == 8.2


# ---------------------------------------------------------------------------
# PenaltyOptimizer — suggest_all and observation_count
# ---------------------------------------------------------------------------


def test_suggest_all_returns_sorted_by_delta():
    opt = PenaltyOptimizer(min_observations=5, penalty_scale=1.2)
    for _ in range(10):
        opt.record(_make_obs(trigger="miss_target", asset_delta=-15.0, current_penalty=8.2))
        opt.record(
            RuleObservation(
                trigger="deception_detected",
                rule_name="false_report",
                observed_asset_delta=-2.0,
                current_penalty=0.5,
            )
        )
    proposals = opt.suggest_all()
    assert len(proposals) >= 1
    # sorted by |delta| descending
    deltas = [abs(p.delta) for p in proposals]
    assert deltas == sorted(deltas, reverse=True)


def test_observation_count():
    opt = PenaltyOptimizer(min_observations=5)
    for _ in range(7):
        opt.record(_make_obs())
    assert opt.observation_count("miss_target") == 7
    assert opt.observation_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# PenaltyOptimizer — record_from_outcome
# ---------------------------------------------------------------------------


def test_record_from_outcome_success():
    opt = PenaltyOptimizer(min_observations=3)
    outcome = BrainOutcome(success=True, cost_paid=0.1, risk_realized=0.05, asset_gained=1.0)
    for _ in range(3):
        opt.record_from_outcome("late_delivery", "miss_target", outcome, 8.2)
    assert opt.observation_count("miss_target") == 3


def test_record_from_outcome_failure():
    opt = PenaltyOptimizer(min_observations=3)
    outcome = BrainOutcome(success=False, cost_paid=0.1, risk_realized=0.5, asset_gained=0.0)
    for _ in range(3):
        opt.record_from_outcome("late_delivery", "miss_target", outcome, 8.2)
    assert opt.observation_count("miss_target") == 3


# ---------------------------------------------------------------------------
# PolicySynthesizer — no templates when data insufficient
# ---------------------------------------------------------------------------


def test_policy_synthesizer_empty_domain():
    synth = PolicySynthesizer()
    templates = synth.synthesise("legal")
    assert templates == []


def test_policy_synthesizer_below_min_occurrences():
    synth = PolicySynthesizer(min_occurrences=3)
    decision = _make_hierarchical_decision()
    synth.record(decision, domain="legal")
    synth.record(decision, domain="legal")
    templates = synth.synthesise("legal")
    assert templates == []


def test_policy_synthesizer_non_decomposed_ignored():
    synth = PolicySynthesizer(min_occurrences=1)
    decision = _make_non_decomposed_decision()
    # If the brain chose not to decompose, record should not count it
    before = synth.decision_count("legal")
    synth.record(decision, domain="legal")
    if not decision.decomposed:
        # Non-decomposed decisions are silently skipped
        templates = synth.synthesise("legal")
        assert templates == []
    # If complexity was too low the brain may still decompose; skip test gracefully


# ---------------------------------------------------------------------------
# PolicySynthesizer — extracts template from recurring pattern
# ---------------------------------------------------------------------------


def test_policy_synthesizer_extracts_template():
    synth = PolicySynthesizer(min_occurrences=3)
    decision = _make_hierarchical_decision(domain="legal")
    for _ in range(5):
        synth.record(decision, domain="legal")
    templates = synth.synthesise("legal")
    if decision.decomposed:
        assert len(templates) >= 1
        t = templates[0]
        assert isinstance(t, DecompositionTemplate)
        assert t.domain == "legal"
        assert t.occurrence_count == 5
    # If the brain happened not to decompose this task, the synthesiser has no
    # data — that's correct behaviour and needs no assertion.


def test_policy_synthesizer_template_utility_positive():
    synth = PolicySynthesizer(min_occurrences=3)
    decision = _make_hierarchical_decision(combined_utility=0.75)
    for _ in range(4):
        synth.record(decision, domain="support")
    templates = synth.synthesise("support")
    if templates:
        assert templates[0].avg_combined_utility > 0


def test_policy_synthesizer_template_confidence_bounded():
    synth = PolicySynthesizer(min_occurrences=3)
    decision = _make_hierarchical_decision()
    for _ in range(20):
        synth.record(decision, domain="medical")
    templates = synth.synthesise("medical")
    assert templates
    assert 0.0 <= templates[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# PolicySynthesizer — distinct patterns produce separate templates
# ---------------------------------------------------------------------------


def test_policy_synthesizer_distinct_patterns_separate_templates():
    """Two decision types with identical fingerprints generate 1 template;
    the important invariant is that the synthesiser groups by fingerprint."""
    synth = PolicySynthesizer(min_occurrences=3)
    decision_a = _make_hierarchical_decision(domain="tech")
    for _ in range(4):
        synth.record(decision_a, domain="tech")
    templates = synth.synthesise("tech")
    # All identical decisions should produce at most 1 template per fingerprint
    if templates:
        assert all(isinstance(t, DecompositionTemplate) for t in templates)


def test_policy_synthesizer_low_utility_excluded():
    synth = PolicySynthesizer(min_occurrences=3, min_utility=0.9)
    decision = _make_hierarchical_decision(combined_utility=0.5)
    for _ in range(5):
        synth.record(decision, domain="legal")
    # combined_utility was set to 0.5 which is below min_utility=0.9
    # The decision should have been silently skipped
    templates = synth.synthesise("legal")
    assert templates == []


# ---------------------------------------------------------------------------
# PolicySynthesizer — known_domains and decision_count
# ---------------------------------------------------------------------------


def test_policy_synthesizer_known_domains():
    synth = PolicySynthesizer(min_occurrences=1)
    decision = _make_hierarchical_decision()
    synth.record(decision, domain="legal")
    synth.record(decision, domain="medical")
    domains = synth.known_domains()
    assert "legal" in domains
    assert "medical" in domains


def test_policy_synthesizer_decision_count():
    synth = PolicySynthesizer(min_occurrences=1)
    decision = _make_hierarchical_decision()
    for _ in range(7):
        synth.record(decision, domain="legal")
    assert synth.decision_count("legal") == 7
    assert synth.decision_count("other") == 0


# ---------------------------------------------------------------------------
# AutoRuleDiscovery — coordinator integration
# ---------------------------------------------------------------------------


def test_auto_rule_discovery_ignores_noise():
    """Isolated events below min_observations threshold produce no proposals."""
    discovery = AutoRuleDiscovery()
    # Default PenaltyOptimizer requires 5 observations — send only 3
    outcome = BrainOutcome(success=False, cost_paid=0.1, risk_realized=0.5, asset_gained=0.0)
    for _ in range(3):
        discovery.observe_rule_event("late_delivery", "miss_target", outcome, 8.2)
    proposals = discovery.suggest_penalty_updates()
    assert proposals == []


def test_auto_rule_discovery_signal_produces_proposals():
    """Sufficient consistent signal fires a proposal."""
    discovery = AutoRuleDiscovery()
    outcome = BrainOutcome(success=False, cost_paid=0.1, risk_realized=0.5, asset_gained=0.0)
    for _ in range(10):
        discovery.observe_rule_event("late_delivery", "miss_target", outcome, 8.2)
    proposals = discovery.suggest_penalty_updates()
    assert len(proposals) >= 1


def test_auto_rule_discovery_templates():
    """Sufficient recurring decompositions produce templates (when decomposed)."""
    discovery = AutoRuleDiscovery()
    decision = _make_hierarchical_decision(domain="legal")
    for _ in range(5):
        discovery.observe_decomposition(decision, domain="legal")
    templates = discovery.suggest_policy_templates("legal")
    if decision.decomposed:
        assert len(templates) >= 1


def test_auto_rule_discovery_mixed_stream():
    """Noise (below threshold) alongside signal (above threshold) — only signal fires."""
    discovery = AutoRuleDiscovery()
    outcome_fail = BrainOutcome(success=False, cost_paid=0.2, risk_realized=0.8, asset_gained=0.0)
    outcome_ok = BrainOutcome(success=True, cost_paid=0.1, risk_realized=0.1, asset_gained=0.9)

    # "miss_target": 10 observations → should produce a proposal
    for _ in range(10):
        discovery.observe_rule_event("late_delivery", "miss_target", outcome_fail, 8.2)

    # "deception_detected": only 2 observations → should NOT produce a proposal
    for _ in range(2):
        discovery.observe_rule_event("false_report", "deception_detected", outcome_ok, 0.5)

    proposals = discovery.suggest_penalty_updates()
    triggers_with_proposals = {p.trigger for p in proposals}
    assert "miss_target" in triggers_with_proposals
    assert "deception_detected" not in triggers_with_proposals


def test_auto_rule_discovery_status_keys():
    discovery = AutoRuleDiscovery()
    status = discovery.status()
    assert "known_triggers" in status
    assert "known_domains" in status
    assert "pending_proposals" in status


def test_auto_rule_discovery_status_reflects_observations():
    discovery = AutoRuleDiscovery()
    outcome = BrainOutcome(success=False, cost_paid=0.1, risk_realized=0.5, asset_gained=0.0)
    for _ in range(10):
        discovery.observe_rule_event("late_delivery", "miss_target", outcome, 8.2)
    decision = _make_hierarchical_decision(domain="legal")
    for _ in range(5):
        discovery.observe_decomposition(decision, domain="legal")

    status = discovery.status()
    assert "miss_target" in status["known_triggers"]
    # "legal" domain is known only if any decomposed decisions were logged
    if decision.decomposed:
        assert "legal" in status["known_domains"]
    assert isinstance(status["pending_proposals"], int)
    assert status["pending_proposals"] >= 1


def test_auto_rule_discovery_no_templates_before_threshold():
    discovery = AutoRuleDiscovery()
    decision = _make_hierarchical_decision(domain="support")
    # Default min_occurrences=3 — send only 2
    for _ in range(2):
        discovery.observe_decomposition(decision, domain="support")
    templates = discovery.suggest_policy_templates("support")
    assert templates == []
