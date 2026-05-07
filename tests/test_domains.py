"""Tests for manifold/domains/ package."""
from __future__ import annotations
import pytest
from manifold.domains import load_domain, available_domains
from manifold.policy import ManifoldPolicy


def test_load_healthcare_returns_policy():
    policy = load_domain("healthcare")
    assert isinstance(policy, ManifoldPolicy)
    assert policy.domains[0].name == "healthcare"


def test_infrastructure_escalation_threshold_low():
    policy = load_domain("infrastructure")
    domain = policy.domains[0]
    assert domain.escalation_threshold < 0.25


def test_infrastructure_no_answer_action():
    policy = load_domain("infrastructure")
    domain = policy.domains[0]
    assert "answer" not in domain.allowed_actions


def test_trading_has_stop_action():
    policy = load_domain("trading")
    domain = policy.domains[0]
    assert "stop" in domain.allowed_actions


def test_unknown_domain_raises_value_error():
    with pytest.raises(ValueError, match="Unknown domain"):
        load_domain("unknown_xyz")


def test_all_7_domains_refusal_less_than_escalation():
    domain_names = ["healthcare", "finance", "devops", "legal", "infrastructure", "trading", "supply_chain"]
    for name in domain_names:
        policy = load_domain(name)
        domain = policy.domains[0]
        assert domain.refusal_threshold < domain.escalation_threshold, (
            f"{name}: refusal_threshold ({domain.refusal_threshold}) "
            f">= escalation_threshold ({domain.escalation_threshold})"
        )


def test_available_domains_returns_list():
    domains = available_domains()
    assert isinstance(domains, list)
    assert len(domains) > 0
