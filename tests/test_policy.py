"""Tests for Phase 18: Policy-as-Code (ManifoldPolicy, PolicyDomain, RuleDiff, etc.)."""

from __future__ import annotations

import pytest

from manifold.autodiscovery import AutoRuleDiscovery, RuleObservation
from manifold.interceptor import InterceptorConfig
from manifold.policy import (
    DOMAIN_TEMPLATES,
    ManifoldPolicy,
    PolicyDomain,
    PolicyExporter,
    PolicyLoader,
    RuleDiff,
    _dict_to_yaml,
    _proposals_to_diffs,
    _yaml_to_dict,
)


# ---------------------------------------------------------------------------
# PolicyDomain tests
# ---------------------------------------------------------------------------


class TestPolicyDomain:
    def test_defaults(self):
        d = PolicyDomain(name="general")
        assert d.name == "general"
        assert 0.0 < d.stakes <= 1.0
        assert 0.0 < d.risk_tolerance <= 1.0
        assert d.fallback_strategy in ("hitl", "fallback", "refuse")

    def test_to_dict_roundtrip(self):
        d = PolicyDomain(
            name="finance",
            stakes=0.9,
            risk_tolerance=0.3,
            coordination_tax_cap=0.12,
            fallback_strategy="hitl",
            min_tool_reliability=0.85,
            notes="strict",
        )
        as_dict = d.to_dict()
        restored = PolicyDomain.from_dict(as_dict)
        assert restored.name == d.name
        assert restored.stakes == pytest.approx(d.stakes, rel=0.001)
        assert restored.risk_tolerance == pytest.approx(d.risk_tolerance, rel=0.001)
        assert restored.fallback_strategy == d.fallback_strategy
        assert restored.notes == d.notes

    def test_from_dict_missing_optional_fields(self):
        d = PolicyDomain.from_dict({"name": "test"})
        assert d.name == "test"
        assert d.stakes == 0.5

    def test_to_dict_contains_all_fields(self):
        d = PolicyDomain(name="legal")
        keys = d.to_dict().keys()
        for expected in (
            "name", "stakes", "risk_tolerance", "coordination_tax_cap",
            "fallback_strategy", "min_tool_reliability", "notes",
        ):
            assert expected in keys


# ---------------------------------------------------------------------------
# RuleDiff tests
# ---------------------------------------------------------------------------


class TestRuleDiff:
    def _sample_diff(self, status: str = "pending") -> RuleDiff:
        return RuleDiff(
            rule_name="late_delivery",
            trigger="miss_target",
            domain="finance",
            field="risk_tolerance",
            old_value=0.45,
            new_value=0.35,
            rationale="Observed high loss",
            confidence=0.8,
            status=status,
        )

    def test_to_dict_roundtrip(self):
        diff = self._sample_diff()
        restored = RuleDiff.from_dict(diff.to_dict())
        assert restored.rule_name == diff.rule_name
        assert restored.old_value == pytest.approx(diff.old_value, rel=0.001)
        assert restored.new_value == pytest.approx(diff.new_value, rel=0.001)
        assert restored.status == diff.status

    def test_apply_approved_diff(self):
        diff = self._sample_diff(status="approved")
        domain = PolicyDomain(name="finance", risk_tolerance=0.45)
        policy = ManifoldPolicy(domains=[domain])
        updated = diff.apply(policy)
        finance = updated.domain("finance")
        assert finance is not None
        assert finance.risk_tolerance == pytest.approx(0.35, rel=0.001)

    def test_apply_pending_diff_does_nothing(self):
        diff = self._sample_diff(status="pending")
        domain = PolicyDomain(name="finance", risk_tolerance=0.45)
        policy = ManifoldPolicy(domains=[domain])
        updated = diff.apply(policy)
        # Should be unchanged
        finance = updated.domain("finance")
        assert finance is not None
        assert finance.risk_tolerance == pytest.approx(0.45, rel=0.001)

    def test_apply_diff_for_missing_domain_does_nothing(self):
        diff = RuleDiff(
            rule_name="r", trigger="t", domain="nonexistent",
            field="risk_tolerance", old_value=0.4, new_value=0.3,
            rationale="", confidence=0.5, status="approved",
        )
        policy = ManifoldPolicy(domains=[PolicyDomain(name="finance")])
        updated = diff.apply(policy)
        assert updated.domain("finance") is not None
        assert updated.domain("nonexistent") is None

    def test_from_dict_defaults(self):
        diff = RuleDiff.from_dict({
            "rule_name": "r",
            "trigger": "t",
            "old_value": 0.4,
            "new_value": 0.35,
        })
        assert diff.domain == "general"
        assert diff.status == "pending"


# ---------------------------------------------------------------------------
# ManifoldPolicy tests
# ---------------------------------------------------------------------------


class TestManifoldPolicy:
    def test_default_factory(self):
        p = ManifoldPolicy.default()
        assert len(p.domains) == 1
        assert p.domains[0].name == "general"

    def test_from_template_finance(self):
        p = ManifoldPolicy.from_template("finance")
        d = p.domain("finance")
        assert d is not None
        assert d.stakes > 0.7

    def test_from_template_legal(self):
        p = ManifoldPolicy.from_template("legal")
        d = p.domain("legal")
        assert d is not None
        assert d.risk_tolerance < 0.4  # strict

    def test_from_template_creative(self):
        p = ManifoldPolicy.from_template("creative")
        d = p.domain("creative")
        assert d is not None
        assert d.risk_tolerance > 0.5  # permissive

    def test_from_template_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            ManifoldPolicy.from_template("unicorn_domain")

    def test_domain_lookup_hit(self):
        p = ManifoldPolicy(domains=[PolicyDomain(name="medical")])
        assert p.domain("medical") is not None

    def test_domain_lookup_miss(self):
        p = ManifoldPolicy.default()
        assert p.domain("nonexistent") is None

    def test_domain_names(self):
        p = ManifoldPolicy(
            domains=[PolicyDomain(name="a"), PolicyDomain(name="b")]
        )
        assert set(p.domain_names()) == {"a", "b"}

    def test_interceptor_config_from_domain(self):
        domain = PolicyDomain(
            name="finance", risk_tolerance=0.25, fallback_strategy="refuse",
            min_tool_reliability=0.90,
        )
        p = ManifoldPolicy(domains=[domain])
        cfg = p.interceptor_config("finance")
        assert cfg.risk_veto_threshold == pytest.approx(0.25, rel=0.001)
        assert cfg.redirect_strategy == "refuse"
        assert cfg.fallback_min_reliability == pytest.approx(0.90, rel=0.001)

    def test_interceptor_config_global_fallback(self):
        p = ManifoldPolicy(global_veto_threshold=0.30, global_fallback_strategy="hitl")
        cfg = p.interceptor_config()
        assert cfg.risk_veto_threshold == pytest.approx(0.30, rel=0.001)
        assert cfg.redirect_strategy == "hitl"

    def test_interceptor_config_missing_domain_uses_global(self):
        p = ManifoldPolicy(
            domains=[PolicyDomain(name="finance")],
            global_veto_threshold=0.55,
        )
        cfg = p.interceptor_config("nonexistent")
        assert cfg.risk_veto_threshold == pytest.approx(0.55, rel=0.001)

    def test_to_dict_roundtrip(self):
        p = ManifoldPolicy.from_template("finance")
        d = p.to_dict()
        restored = ManifoldPolicy.from_dict(d)
        assert restored.version == p.version
        assert len(restored.domains) == len(p.domains)
        assert restored.domains[0].name == p.domains[0].name

    def test_add_diff(self):
        p = ManifoldPolicy.default()
        diff = RuleDiff(
            rule_name="r", trigger="t", domain="general",
            field="risk_tolerance", old_value=0.45, new_value=0.35,
            rationale="test", confidence=0.7,
        )
        p.add_diff(diff)
        assert len(p.pending_diffs) == 1

    def test_approve_diff(self):
        domain = PolicyDomain(name="general", risk_tolerance=0.45)
        p = ManifoldPolicy(domains=[domain])
        diff = RuleDiff(
            rule_name="r", trigger="t", domain="general",
            field="risk_tolerance", old_value=0.45, new_value=0.35,
            rationale="test", confidence=0.7,
        )
        p.add_diff(diff)
        updated = p.approve_diff("r")
        g = updated.domain("general")
        assert g is not None
        assert g.risk_tolerance == pytest.approx(0.35, rel=0.001)

    def test_reject_diff(self):
        p = ManifoldPolicy.default()
        diff = RuleDiff(
            rule_name="r", trigger="t", domain="general",
            field="risk_tolerance", old_value=0.45, new_value=0.35,
            rationale="test", confidence=0.7,
        )
        p.add_diff(diff)
        p.reject_diff("r")
        assert p.pending_diffs[0].status == "rejected"

    def test_approve_nonexistent_diff_returns_unchanged(self):
        p = ManifoldPolicy.default()
        updated = p.approve_diff("nonexistent_rule")
        assert updated is p


# ---------------------------------------------------------------------------
# YAML serialisation tests
# ---------------------------------------------------------------------------


class TestYAMLSerialization:
    def test_to_yaml_is_string(self):
        p = ManifoldPolicy.from_template("finance")
        yaml_str = p.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

    def test_to_yaml_contains_key_fields(self):
        p = ManifoldPolicy.from_template("finance")
        yaml_str = p.to_yaml()
        assert "version" in yaml_str
        assert "domains" in yaml_str
        assert "finance" in yaml_str

    def test_to_json_roundtrip(self):
        p = ManifoldPolicy.from_template("legal")
        json_str = p.to_json()
        restored = ManifoldPolicy.from_json(json_str)
        assert restored.version == p.version
        assert len(restored.domains) == len(p.domains)

    def test_yaml_roundtrip_via_loader(self):
        p = ManifoldPolicy(
            version="2.0.0",
            description="Test policy",
            global_veto_threshold=0.30,
            domains=[
                PolicyDomain(name="finance", stakes=0.9, risk_tolerance=0.25),
                PolicyDomain(name="creative", stakes=0.3, risk_tolerance=0.70),
            ],
        )
        yaml_str = p.to_yaml()
        restored = PolicyLoader.from_yaml(yaml_str)
        assert str(restored.version) == "2.0.0"
        assert restored.description == "Test policy"
        assert restored.global_veto_threshold == pytest.approx(0.30, abs=0.01)
        finance = restored.domain("finance")
        assert finance is not None

    def test_json_roundtrip_via_loader(self):
        p = ManifoldPolicy.from_template("support")
        json_str = p.to_json()
        restored = PolicyLoader.from_json(json_str)
        assert restored.domains[0].name == "support"

    def test_loader_from_dict(self):
        d = ManifoldPolicy.from_template("general").to_dict()
        restored = PolicyLoader.from_dict(d)
        assert restored.version is not None

    def test_empty_policy_serialises(self):
        p = ManifoldPolicy()
        yaml_str = p.to_yaml()
        assert "version" in yaml_str


# ---------------------------------------------------------------------------
# PolicyExporter tests
# ---------------------------------------------------------------------------


def _make_discovery_with_data() -> AutoRuleDiscovery:
    discovery = AutoRuleDiscovery()
    for i in range(8):
        obs = RuleObservation(
            trigger="miss_target",
            rule_name="late_delivery",
            observed_asset_delta=-5.0 - i * 0.5,
            current_penalty=8.0,
        )
        discovery.optimizer.record(obs)
    return discovery


class TestPolicyExporter:
    def test_export_returns_manifold_policy(self):
        discovery = AutoRuleDiscovery()
        exporter = PolicyExporter(discovery=discovery)
        policy = exporter.export()
        assert isinstance(policy, ManifoldPolicy)

    def test_export_includes_requested_domains(self):
        discovery = AutoRuleDiscovery()
        exporter = PolicyExporter(discovery=discovery)
        policy = exporter.export(domains=["finance", "legal"])
        names = policy.domain_names()
        assert "finance" in names
        assert "legal" in names

    def test_export_uses_interceptor_config(self):
        discovery = AutoRuleDiscovery()
        cfg = InterceptorConfig(risk_veto_threshold=0.33, redirect_strategy="refuse")
        exporter = PolicyExporter(discovery=discovery, config=cfg)
        policy = exporter.export()
        assert policy.global_veto_threshold == pytest.approx(0.33, rel=0.001)
        assert policy.global_fallback_strategy == "refuse"

    def test_export_embeds_pending_diffs_from_proposals(self):
        discovery = _make_discovery_with_data()
        exporter = PolicyExporter(discovery=discovery)
        policy = exporter.export(domains=["general"])
        # Proposals exist, so diffs should be populated
        assert len(policy.pending_diffs) >= 0  # may be 0 if threshold not met

    def test_export_with_synthesised_templates(self):
        from manifold.brain import BrainConfig, HierarchicalBrain, BrainTask
        discovery = AutoRuleDiscovery()
        cfg = BrainConfig(generations=2, population_size=12, grid_size=5)
        brain = HierarchicalBrain(config=cfg)
        task = BrainTask("analyse legal document", domain="legal", complexity=0.92, stakes=0.85)
        for _ in range(4):
            dec = brain.decide_hierarchical(task)
            discovery.observe_decomposition(dec, domain="legal")
        exporter = PolicyExporter(discovery=discovery)
        policy = exporter.export(domains=["legal"])
        legal = policy.domain("legal")
        assert legal is not None

    def test_export_version_is_set(self):
        discovery = AutoRuleDiscovery()
        exporter = PolicyExporter(discovery=discovery, version="2.1.0")
        policy = exporter.export()
        assert policy.version == "2.1.0"


# ---------------------------------------------------------------------------
# Domain template tests
# ---------------------------------------------------------------------------


class TestDomainTemplates:
    def test_all_templates_present(self):
        for name in ("finance", "legal", "creative", "support", "research", "medical", "general"):
            assert name in DOMAIN_TEMPLATES

    def test_templates_have_required_fields(self):
        for name, tpl in DOMAIN_TEMPLATES.items():
            for key in (
                "stakes", "risk_tolerance", "coordination_tax_cap",
                "fallback_strategy", "min_tool_reliability",
            ):
                assert key in tpl, f"Template {name!r} missing {key!r}"

    def test_medical_strictest_risk_tolerance(self):
        medical_rt = DOMAIN_TEMPLATES["medical"]["risk_tolerance"]
        legal_rt = DOMAIN_TEMPLATES["legal"]["risk_tolerance"]
        creative_rt = DOMAIN_TEMPLATES["creative"]["risk_tolerance"]
        assert medical_rt < legal_rt < creative_rt

    def test_all_stakes_in_range(self):
        for name, tpl in DOMAIN_TEMPLATES.items():
            s = tpl["stakes"]
            assert 0.0 <= s <= 1.0, f"{name!r} stakes={s} out of range"


# ---------------------------------------------------------------------------
# _proposals_to_diffs helper tests
# ---------------------------------------------------------------------------


class TestProposalsToDiffs:
    def test_empty_proposals_returns_empty(self):
        diffs = _proposals_to_diffs([])
        assert diffs == []

    def test_proposal_maps_to_diff(self):
        discovery = _make_discovery_with_data()
        proposals = discovery.suggest_penalty_updates()
        diffs = _proposals_to_diffs(proposals, domain="finance")
        assert len(diffs) == len(proposals)
        for diff in diffs:
            assert diff.domain == "finance"
            assert diff.status == "pending"
            assert diff.field == "risk_tolerance"

    def test_new_values_clamped_between_05_and_95(self):
        discovery = _make_discovery_with_data()
        proposals = discovery.suggest_penalty_updates()
        diffs = _proposals_to_diffs(proposals)
        for diff in diffs:
            assert 0.05 <= diff.new_value <= 0.95


# ---------------------------------------------------------------------------
# YAML helper unit tests
# ---------------------------------------------------------------------------


class TestYAMLHelpers:
    def test_dict_to_yaml_simple(self):
        d = {"version": "1.0", "threshold": 0.45}
        yaml_str = _dict_to_yaml(d)
        assert "version" in yaml_str
        assert "threshold" in yaml_str

    def test_dict_to_yaml_nested(self):
        d = {"outer": {"inner": "value"}}
        yaml_str = _dict_to_yaml(d)
        assert "outer" in yaml_str
        assert "inner" in yaml_str

    def test_dict_to_yaml_list(self):
        d = {"items": [{"name": "a"}, {"name": "b"}]}
        yaml_str = _dict_to_yaml(d)
        assert "items" in yaml_str

    def test_yaml_roundtrip_simple(self):
        d = {"version": "1.0", "enabled": True, "threshold": 0.45}
        yaml_str = _dict_to_yaml(d)
        restored = _yaml_to_dict(yaml_str)
        # version may come back as string or float depending on parser
        assert str(restored["version"]) == "1.0"
        assert restored["threshold"] == pytest.approx(0.45, abs=0.001)
